"""Incremental pose update vs full reload: short GPU comparison.

Order of truth: OUTPUT CONSISTENCY FIRST (per-frame RGB must match between the
two paths), performance second. Metrics per path: preparation time, per-frame
commit stall, first-frame latency, peak VRAM.

  Path R (full reload):  edit(remove) + edit(insert) per frame -> gs.pth is
                         re-read and the model rebuilt every frame.
  Path I (incremental):  insert once, then advance_pose_track per frame
                         (in-place SH rotation + pose tensor).

Diagnostic mode only; no UniAD, no control; does not overwrite anything.
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

pass  # Path is provided by the portable boundary above
import sys, json, time
import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
P11 = JOB/'chain_repair/phase11'
for p in (str(HERE), str(P11), str(ROOT/'integrations/sr-hugsim'),
          str(ROOT/'waymo23-native-import')):
    sys.path.insert(0, p)

import drivable_area  # noqa: F401
import road_route_seed  # noqa: F401
from scene_api import SceneSession
from pose_track import PoseTrajectory
from session_adapter import PoseTrackSession
from gpu_applier import make_backend_pose_applier

OUT = HERE/'compare_evidence'
FPS = 10.0
START_FRAME = 80
STEPS = 10


def fresh_session():
    manifest = _hugsim_expand(json.loads((JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json').read_text()))
    manifest['render_scale'] = 1.
    (OUT/'scene_manifest.json').write_text(json.dumps(manifest, indent=2))
    s = SceneSession()
    s.load_scene(OUT/'scene_manifest.json')
    return s


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    import torch

    insert = _hugsim_expand(json.loads((BRIDGE/'requests/closed552_insert_realcar.json').read_text()))
    anchor = np.asarray(insert['asset_to_world'], dtype=float)
    times = 8.0 + np.arange(STEPS) / FPS
    key_poses = []
    for t in times:
        p = anchor.copy(); p[:3, 3] = anchor[:3, 3] + anchor[:3, 0] * (2.0 * (t - 8.0))
        key_poses.append(p)
    trajectory = PoseTrajectory(times, key_poses, bounds='strict')

    session = fresh_session()
    spec = session.spec
    inputs = Path(spec.source['inputs']); raw0 = spec.source['start_timestep']
    poses = np.stack([np.loadtxt(inputs/'ego_pose'/f'{raw0+i:03d}.txt') for i in range(spec.frames)])
    relative = np.linalg.inv(poses[0])[None] @ poses

    report = {}

    # ---- Path R: full reload per frame ----
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    frames_r, stalls_r = [], []
    first_latency = None
    for k, t in enumerate(times):
        req = dict(insert); req['asset_to_world'] = key_poses[k].tolist()
        c0 = time.perf_counter()
        if k:
            session.edit({'operation': 'remove', 'object_id': 'preview_car', 'mode': 'diagnostic_preview'})
        session.edit(req)
        c1 = time.perf_counter()
        packet = session.render(relative[START_FRAME + k], float(t))
        c2 = time.perf_counter()
        if k == 0:
            first_latency = c2 - c0
        stalls_r.append(c1 - c0)
        frames_r.append(packet['cameras']['CAM_FRONT']['rgb'].copy())
    report['reload'] = dict(prepare_s=None, first_frame_latency_s=first_latency,
                            commit_stall_s=stalls_r, total_s=time.perf_counter() - t0,
                            peak_vram_mb=torch.cuda.max_memory_allocated()/1e6)
    session.reset()

    # ---- Path I: incremental pose track ----
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    p0 = time.perf_counter()
    adapter = PoseTrackSession(session, make_backend_pose_applier(session.backend))
    adapter.insert_tracked('preview_car', trajectory, edit_request=insert)
    prepare = time.perf_counter() - p0
    frames_i, stalls_i = [], []
    first_latency = None
    for k, t in enumerate(times):
        c0 = time.perf_counter()
        adapter.advance_pose_track(float(t))
        c1 = time.perf_counter()
        packet = session.render(relative[START_FRAME + k], float(t))
        c2 = time.perf_counter()
        if k == 0:
            first_latency = c2 - c0
        stalls_i.append(c1 - c0)
        frames_i.append(packet['cameras']['CAM_FRONT']['rgb'].copy())
    report['incremental'] = dict(prepare_s=prepare, first_frame_latency_s=first_latency,
                                 commit_stall_s=stalls_i, total_s=time.perf_counter() - t0,
                                 peak_vram_mb=torch.cuda.max_memory_allocated()/1e6)

    # ---- Output consistency first ----
    max_abs = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(frames_r, frames_i))
    identical = all(np.array_equal(a, b) for a, b in zip(frames_r, frames_i))
    report['output_consistency'] = dict(pixel_identical=bool(identical), max_abs_uint8_diff=max_abs,
                                        frames=STEPS)
    report['scope'] = ('consistency first, performance second; render-only; '
                       'diagnostic uncertified; no UniAD; no control')
    (OUT/'compare_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('CONSISTENCY:', 'PIXEL-IDENTICAL' if identical else f'max abs diff {max_abs}')


if __name__ == '__main__':
    main()
