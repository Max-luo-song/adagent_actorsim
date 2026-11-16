"""Short executed-pose weather integration using the existing framework harness."""

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

import argparse
import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0, str(BRIDGE / 'framework_backend'))
import run_arm_fwbackend as fw
import scene_api_fwbackend as saf
from runtime_backend import RuntimeWeatherBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    request = _hugsim_expand(json.loads((BRIDGE / 'framework_backend/requests/fastprobe_fwbackend.json').read_text()))
    request.update(stop_on_stationary=False, complete_on_pass=False)
    request_path = out / 'weather_request.json'
    request_path.write_text(json.dumps(request, indent=2))
    sources = list(HERE.glob('*.py')) + [
        BRIDGE / 'weather_sequence/causal_buffer.py',
        BRIDGE / 'framework_backend/backend.py',
        BRIDGE / 'framework_backend/run_arm_fwbackend.py',
        BRIDGE / 'unified/run_phase26.sh']
    (out / 'implementation.json').write_text(json.dumps({str(f): hashlib.sha256(
        f.read_bytes()).hexdigest() for f in sources}, indent=2))
    records, packets = [], []

    class WeatherSession(saf.SceneSession):
        def __init__(self):
            super().__init__(backend_factory=lambda spec: RuntimeWeatherBackend(spec, out / 'weather'))

        def _packet(self, raw, timestamp, ego_pose):
            packet = super()._packet(raw, timestamp, ego_pose)
            packets.append(dict(timestamp=float(timestamp),
                rgb_sha256={name: hashlib.sha256(c['rgb'].tobytes()).hexdigest()
                            for name, c in packet['cameras'].items()}))
            return packet

        def close(self):
            if self.backend is not None:
                records[:] = self.backend.weather_records
            super().close()

    import paired_closed_loop_arm as harness
    original = harness.SceneSession
    harness.SceneSession = WeatherSession
    try:
        fw.run(str(request_path), str(out / 'closed_loop'))
    finally:
        harness.SceneSession = original
        (out / 'weather_packet_audit.json').write_text(json.dumps(packets, indent=2))
    report = _hugsim_expand(json.loads((out / 'closed_loop/run_report.json').read_text()))
    observations = report.get('observations', [])
    checks = dict(
        completed=report.get('status') == 'completed',
        five_executed_controls=len(report.get('trace', [])) == 5,
        six_weather_observations=len(records) == len(observations) == len(packets) == 6,
        control_feedback=report.get('control_actions_executed') is True,
        no_recorded_future=report.get('recorded_future_ego_used_for_closed_motion') is False,
        no_formal_admission=report.get('formal_evaluation_admitted') is False,
        executed_pose_matches_weather=bool(records) and len(records) == len(observations) and all(
            np.allclose(r['ego_to_world'], o['closed_ego_to_world'], atol=1e-6, rtol=0)
            and abs(r['timestamp'] - o['timestamp']) < 1e-6 for r, o in zip(records, observations)),
        historical_only=bool(records) and all(not r['audit']['future_frames_used'] and
            max(r['audit']['window_timestamps']) <= r['timestamp'] for r in records),
        cameras_aligned=bool(records) and all(np.allclose(
            np.asarray(r['ego_to_world']) @ np.asarray(c['camera_to_ego']),
            c['camera_to_world'], atol=1e-6, rtol=0) for r in records for c in r['cameras'].values()),
        dynamic_rgb_preserved=bool(records) and all(c['actor_rgb_preserved']
            for r in records for c in r['cameras'].values()),
        geometry_depth_unchanged=bool(records) and all(c['depth_unchanged']
            for r in records for c in r['cameras'].values()),
        weather_changes_visible=bool(records) and all(any(c['changed_pixels'] > 0
            for c in r['cameras'].values()) for r in records),
        packet_contains_weather=len(records) == len(packets) and bool(records) and all(
            all(c['rgb_sha256'] == p['rgb_sha256'][name] for name, c in r['cameras'].items())
            for r, p in zip(records, packets)))
    # Check saved model inputs, not merely renderer output or a composed video.
    worker_inputs = []
    for i, record in enumerate(records[:5]):
        path = out / f'closed_loop/uniad_worker/step_{i:03d}/observation.pkl'
        if not path.exists():
            worker_inputs.append(False)
            continue
        with path.open('rb') as f:
            obs, _ = pickle.load(f)
        worker_inputs.append(all(hashlib.sha256(obs['rgb'][name].tobytes()).hexdigest()
            == c['rgb_sha256'] for name, c in record['cameras'].items()))
    checks['uniad_inputs_are_weather_rgb'] = len(worker_inputs) == 5 and all(worker_inputs)
    acceptance = dict(checks=checks, passed=all(checks.values()),
        formal_evaluation_admitted=False, diagnostic_uncertified=True,
        safety_equivalence_to_mpc=False, scope='unedited frame80; 5 iLQR control steps; rainy RGB style',
        temporal_quality_certified=False, physical_weather_geometry=False)
    (out / 'acceptance.json').write_text(json.dumps(acceptance, indent=2))
    if not acceptance['passed']:
        raise SystemExit('weather integration acceptance failed; inspect acceptance.json')


if __name__ == '__main__':
    main()
