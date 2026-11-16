"""Generate the pose_track closed-loop request + tracked insert JSON.

Trajectory: anchor-local +x at 2 m/s (same synthetic family as the accepted
gpu_validate run), strict bounds [t0, t0+0.6]s covering the 5-step window
[8.0, 8.5]s (start_frame=80, fps=10). Deterministic; refuses to overwrite.
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

import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent

START_FRAME = 80
FPS = 10.0
SPEED = 2.0        # m/s, anchor-local +x
WINDOW_S = 0.6     # covers first render (8.0) through step 5 render (8.5)


def main():
    insert = _hugsim_expand(json.loads((BRIDGE / 'requests/closed552_insert_realcar.json').read_text()))
    anchor = np.asarray(insert['asset_to_world'], dtype=float).reshape(4, 4)
    t0 = START_FRAME / FPS
    times = [t0, t0 + WINDOW_S]
    poses = []
    for t in times:
        p = anchor.copy()
        p[:3, 3] = anchor[:3, 3] + anchor[:3, 0] * (SPEED * (t - t0))
        poses.append(p.tolist())
    tracked = dict(insert, pose_track_trajectory=dict(
        times_s=times, asset_to_world_4x4=poses, bounds='strict',
        note='anchor-local +x at 2 m/s; scene_session_world frame; strict window'))
    out_insert = HERE / 'requests/closed552_insert_realcar_track.json'
    out_req = HERE / 'requests/fastprobe_A5_posetrack.json'
    for p in (out_insert, out_req):
        if p.exists():
            raise SystemExit(f'refusing to overwrite: {p}')
    out_insert.write_text(json.dumps(tracked, indent=2))
    req = _hugsim_expand(json.loads((BRIDGE / 'fast_control/requests/fastprobe_A5_tracking.json').read_text()))
    req['insert_request'] = str(out_insert)
    out_req.write_text(json.dumps(req, indent=2))
    print('wrote', out_insert)
    print('wrote', out_req)


if __name__ == '__main__':
    main()
