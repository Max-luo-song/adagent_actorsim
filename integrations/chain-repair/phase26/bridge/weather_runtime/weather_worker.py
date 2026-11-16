"""Persistent WeatherEdit worker in its isolated venv; JSON-lines + npz IPC.

Only current RGB/mask arrives per call. History belongs to this worker; it never
reads recorded future frames. Stdout is reserved for protocol replies.
"""

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

import contextlib
import json
import os
pass  # Path is provided by the portable boundary above
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
FRAMEWORK = BRIDGE.parent / 'framework_repo'
sys.path.insert(0, str(BRIDGE / 'weather_sequence'))
from causal_buffer import CausalWeatherBuffer


STARTED = time.monotonic()


def startup_mark(stage, **details):
    value = dict(stage=stage, elapsed_seconds=time.monotonic() - STARTED,
                 pid=os.getpid(), **details)
    path = os.environ.get('PHASE26_WEATHER_STATUS_FILE')
    if path:
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(target)
    print('Weather startup: ' + json.dumps(value), file=sys.stderr, flush=True)


def send(value):
    sys.__stdout__.write(json.dumps(value) + '\n')
    sys.__stdout__.flush()


def main():
    with contextlib.redirect_stdout(sys.stderr):
        startup_mark('import_torch')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        free, total = torch.cuda.mem_get_info()
        startup_mark('import_weather_model', cuda_free_bytes=free, cuda_total_bytes=total)
        sys.path.insert(0, str(FRAMEWORK / 'WeatherEdit/background_editing/src'))
        os.chdir(FRAMEWORK)
        from cyclegan_turbo import CycleGAN_Turbo
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        checkpoint = FRAMEWORK / 'WeatherEdit/background_editing/ckpts/with_seg.pkl'
        startup_mark('construct_and_load_weather_model', checkpoint=str(checkpoint),
                     checkpoint_bytes=checkpoint.stat().st_size)
        model = CycleGAN_Turbo(3, 3, weather_type='rainy', with_seg=True, tv_att=True,
            pretrained_path=str(checkpoint))
        startup_mark('configure_inference')
        model.eval()
        model.requires_grad_(False)
        rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()

        def predict(images, masks):
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            x = torch.from_numpy(images).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1
            labels = torch.from_numpy(masks)[:, None].float().cuda() / 255
            with torch.inference_mode():
                output = model(x, labels, 0, 9, mode='inference', direction='a2b',
                    caption='Picture of a rainy weather scene', with_seg=True)
            value = output.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
            if value.shape != (9, 256, 384, 3) or not np.isfinite(value).all():
                raise ValueError('invalid model result')
            return np.clip((value * .5 + .5) * 255, 0, 255).astype(np.uint8)

        buffer = CausalWeatherBuffer()
        startup_mark('ready', cuda_allocated_bytes=torch.cuda.memory_allocated(),
                     cuda_reserved_bytes=torch.cuda.memory_reserved())
        send(dict(ready=True, pid=os.getpid(), weather='rainy'))
        for line in sys.stdin:
            try:
                request = _hugsim_expand(json.loads(line))
                if request['operation'] == 'close':
                    send(dict(closed=True))
                    return
                if request['operation'] == 'reset':
                    buffer = CausalWeatherBuffer()
                    send(dict(reset=True))
                    continue
                if request['operation'] != 'predict':
                    raise ValueError('unknown operation')
                began = time.monotonic()
                with np.load(request['input'], allow_pickle=False) as data:
                    images, masks = data['rgb'], data['mask']
                result, audit = buffer.process(request['frame'], request['timestamp'],
                    request['epoch'], images, masks, predict)
                np.save(request['output'], result)
                send(dict(ok=True, audit=audit, elapsed_s=time.monotonic() - began))
            except Exception as exc:
                # Fail closed: no clear-weather fallback on an inference failure.
                send(dict(ok=False, error=f'{type(exc).__name__}: {exc}'))
                return


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        startup_mark('failed', error=f'{type(exc).__name__}: {exc}')
        raise
