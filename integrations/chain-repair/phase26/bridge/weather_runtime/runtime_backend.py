"""Executed-pose framework render -> dynamic mask -> persistent causal weather."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import hashlib
import json
import math
import os
pass  # Path is provided by the portable boundary above
import selectors
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0, str(HERE))
from mask_backend import MaskFrameworkBackend
from runtime_contract import CAMERAS, prepare_triplet, integrate_triplet


class WeatherClient:
    def __init__(self, folder):
        startup_timeout = float(os.environ.get('PHASE26_WEATHER_STARTUP_TIMEOUT', '900'))
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('PHASE26_WEATHER_STARTUP_TIMEOUT must be finite and positive')
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.log = (self.folder / 'worker.log').open('w')
        self.startup_status = self.folder / 'startup_status.json'
        self.process = None
        try:
            python = BRIDGE / 'env_fix/weather_env_v1/bin/python'
            self.process = subprocess.Popen([str(python), '-u', '-B', str(HERE / 'weather_worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                env=dict(os.environ, PHASE26_WEATHER_STATUS_FILE=str(self.startup_status)))
            if not self.receive(timeout=startup_timeout, phase='startup').get('ready'):
                raise RuntimeError('weather worker not ready')
        except Exception:
            self.close()
            raise

    def receive(self, timeout=300, phase='request'):
        began = time.monotonic()
        deadline = began + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = self._startup_snapshot()
                    failure = dict(phase=phase, timeout_seconds=timeout,
                        elapsed_seconds=time.monotonic() - began,
                        worker_returncode=self.process.poll(), last_startup_status=status)
                    (self.folder / 'timeout.json').write_text(json.dumps(failure, indent=2))
                    raise TimeoutError(f'weather worker {phase} timed out after {timeout:g}s; '
                        f'last stage={status.get("stage", "unknown")}; inspect {self.folder / "worker.log"}')
                if selector.select(timeout=min(15., remaining)):
                    break
                if phase == 'startup':
                    status = self._startup_snapshot()
                    print(f'Weather startup: {time.monotonic() - began:.0f}/{timeout:g}s; '
                        f'stage={status.get("stage", "waiting_for_worker")}; log: {self.folder / "worker.log"}',
                        file=sys.stderr, flush=True)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'weather worker exited during {phase}; '
                f'code={self.process.poll()}; inspect {self.folder / "worker.log"}')
        return _hugsim_expand(json.loads(line))

    def _startup_snapshot(self):
        try:
            return _hugsim_expand(json.loads(self.startup_status.read_text()))
        except (OSError, ValueError):
            return {}

    def request(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()
        reply = self.receive()
        if reply.get('ok') is False:
            raise RuntimeError(reply['error'])
        return reply

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None
        self.log.close()


class RuntimeWeatherBackend(MaskFrameworkBackend):
    def __init__(self, spec, output, protect_signals=False):
        self.client = None
        self.weather_records = []
        self.weather_epoch = 0
        self.weather_output = Path(output).resolve()
        self.weather_output.mkdir(parents=True, exist_ok=False)
        super().__init__(spec,protect_signals=protect_signals)
        try:
            self.client = WeatherClient(self.weather_output / 'worker')
        except Exception:
            super().close()
            raise

    def render(self, pose, timestamp):
        began = time.monotonic()
        clear = super().render(pose, timestamp)
        images, masks = prepare_triplet(clear)
        index = len(self.weather_records)
        folder = self.weather_output / f'step_{index:03d}'
        folder.mkdir(exist_ok=False)
        inp, out = folder / 'input.npz', folder / 'prediction.npy'
        np.savez(inp, rgb=images, mask=masks)
        frame = int(getattr(self,'_actor_source_frame',round(float(timestamp) * self.spec.fps)))
        reply = self.client.request(dict(operation='predict', input=str(inp), output=str(out),
            frame=frame, timestamp=float(timestamp), epoch=str(self.weather_epoch)))
        result = integrate_triplet(clear, np.load(out, allow_pickle=False))
        record = dict(index=index, timestamp=float(timestamp), ego_to_world=np.asarray(pose).tolist(),
            managed_objects=sorted(self.managed_assets.assets),
            managed_gaussians_in_last_collection=getattr(self.trainer, 'managed_last_point_count', 0),
            **reply, elapsed_total_s=time.monotonic() - began, cameras={})
        for name in CAMERAS:
            original, current = clear['cameras'][name], result['cameras'][name]
            mask = original['weather_dynamic_mask'] > 0
            Image.fromarray(original['rgb']).save(folder / f'{name}_clear.png')
            Image.fromarray(original['weather_dynamic_mask']).save(folder / f'{name}_mask.png')
            Image.fromarray(current['rgb']).save(folder / f'{name}_weather.png')
            record['cameras'][name] = dict(
                clear_sha256=hashlib.sha256(original['rgb'].tobytes()).hexdigest(),
                rgb_sha256=hashlib.sha256(current['rgb'].tobytes()).hexdigest(),
                actor_pixels=int(mask.sum()), actor_rgb_preserved=bool(np.array_equal(
                    original['rgb'][mask], current['rgb'][mask])),
                changed_pixels=int(np.any(original['rgb'] != current['rgb'], -1).sum()),
                depth_unchanged=bool(np.array_equal(original['depth'], current['depth'])),
                camera_to_world=np.asarray(current['camera_to_world']).tolist(),
                camera_to_ego=np.asarray(self.cameras_data[current['source_camera_id']].cam_to_ego).tolist())
        self.weather_records.append(record)
        self.save_report()
        return result

    def save_report(self):
        path = self.weather_output / 'report.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(records=self.weather_records,
            formal_evaluation_admitted=False, diagnostic_uncertified=True,
            weather='rainy', effect='RGB background style; visible dynamic pixels copied from clear render',
            physical_weather_geometry=False, temporal_quality_certified=False,
            model_input_size=[384, 256],
            sensor_output_size='restored to original render dimensions; original K retained',
            camera_order=list(CAMERAS)), indent=2))
        temporary.replace(path)

    def begin_weather_episode(self,output):
        """Rebind persisted frame outputs while retaining this loaded worker."""
        folder=Path(output).resolve()
        if folder.exists():raise FileExistsError(folder)
        if self._last is not None:raise RuntimeError('Reset the scene before starting another weather episode')
        self.clear_observation_history()
        folder.mkdir(parents=True,exist_ok=False)
        self.weather_output=folder
        self.weather_records=[]
        self.save_report()

    def clear_observation_history(self):
        super().clear_observation_history()
        self.weather_epoch += 1
        if self.client is not None:
            reply = self.client.request(dict(operation='reset'))
            self.last_weather_reset_verified = reply.get('reset') is True
            if not self.last_weather_reset_verified:
                raise RuntimeError('weather worker did not acknowledge history reset')

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        super().close()
