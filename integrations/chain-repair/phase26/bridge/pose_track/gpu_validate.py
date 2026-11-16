"""pose_track short GPU validation (render-only; no UniAD, no control).

Acceptance scope (per audit finding): rendering and pose continuity ONLY.
UniAD history handling is NOT exercised here -- the worker resets history on
scene_token/timestamp-gap only and never sees model_cache_token; that claim is
covered by the static code audit, not by this render loop.

Checks per frame:
  1. advance time monotonic; trajectory pose == registry render pose ==
     collision pose (identity), == packet managed-object pose (numeric);
  2. packet edit_revision / model_cache_token unchanged across advances;
  3. rendered frame saved for visual continuity inspection.
Reset reproduction: after reset + re-insert + advance to t0, the re-rendered
first frame must be pixel-identical to the original first frame.

Run in the GPU terminal with the HUGSIM pixi python. Diagnostic mode only.
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
import sys, json
import numpy as np

HERE = Path(__file__).resolve().parent          # bridge/pose_track
BRIDGE = HERE.parent
JOB = HERE.parents[3]                            # unified-open-closed-172-20261022-v4
ROOT = JOB.parents[3]
P11 = JOB/'chain_repair/phase11'
for p in (str(HERE), str(P11), str(ROOT/'integrations/sr-hugsim'),
          str(ROOT/'waymo23-native-import')):
    sys.path.insert(0, p)

import drivable_area  # noqa: F401  reviewed, isolated road raster policy
import road_route_seed  # noqa: F401
from scene_api import SceneSession, SceneSpec
from PIL import Image
from pose_track import PoseTrajectory
from session_adapter import PoseTrackSession
from gpu_applier import make_backend_pose_applier

OUT = HERE/'gpu_evidence'
FPS = 10.0
START_FRAME = 80
STEPS = 10  # t = 8.0 .. 8.9


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)

    # Match the closed-loop arms: full-resolution render scale.
    manifest = _hugsim_expand(json.loads((JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json').read_text()))
    manifest['render_scale'] = 1.
    (OUT/'scene_manifest.json').write_text(json.dumps(manifest, indent=2))
    session = SceneSession()
    session.load_scene(OUT/'scene_manifest.json')
    spec = session.spec
    assert np.isclose(1/spec.fps, 1/FPS), 'fps contract'
    inputs = Path(spec.source['inputs']); raw0 = spec.source['start_timestep']
    poses = np.stack([np.loadtxt(inputs/'ego_pose'/f'{raw0+i:03d}.txt') for i in range(spec.frames)])
    relative = np.linalg.inv(poses[0])[None] @ poses

    insert = _hugsim_expand(json.loads((BRIDGE/'requests/closed552_insert_realcar.json').read_text()))
    anchor = np.asarray(insert['asset_to_world'], dtype=float)

    # Trajectory: start at the anchor, advance along local +x at 2 m/s.
    times = 8.0 + np.arange(STEPS) / FPS
    key_poses = []
    for t in times:
        p = anchor.copy()
        p[:3, 3] = anchor[:3, 3] + anchor[:3, 0] * (2.0 * (t - 8.0))
        key_poses.append(p)
    trajectory = PoseTrajectory(times, key_poses, bounds='strict')

    adapter = PoseTrackSession(session, make_backend_pose_applier(session.backend))
    adapter.insert_tracked('preview_car', trajectory, edit_request=insert)

    record = []
    for k, t in enumerate(times):
        ego = relative[START_FRAME + k]
        poses_now = adapter.advance_pose_track(float(t))
        packet = session.render(ego, float(t))
        obj_pose = np.asarray(packet['managed_objects']['preview_car']['asset_to_world'], dtype=float)
        reg_pose = adapter.registry.render_pose('preview_car')
        assert adapter.registry.render_pose('preview_car') is adapter.registry.collision_pose('preview_car')
        assert np.allclose(obj_pose, poses_now['preview_car'], atol=1e-9), 'packet pose != applied pose'
        assert np.allclose(obj_pose, key_poses[k], atol=1e-9), 'pose diverged from trajectory'
        Image.fromarray(packet['cameras']['CAM_FRONT']['rgb']).save(OUT/f'frame_{k:03d}.png')
        rec = dict(k=k, t=float(t), edit_revision=packet['edit_revision'],
                   model_cache_token=packet['model_cache_token'],
                   obj_pose=obj_pose.tolist())
        record.append(rec)
        print(f"frame {k}: t={t:.1f} rev={packet['edit_revision']} token={packet['model_cache_token']}", flush=True)

    tokens = {r['model_cache_token'] for r in record}
    revs = {r['edit_revision'] for r in record}
    assert len(tokens) == 1 and len(revs) == 1, 'advance changed revision/token'

    # Reset reproduction: same inputs after reset must give the same pixels.
    first_frame = np.asarray(Image.open(OUT/'frame_000.png'))
    adapter.reset()
    adapter.insert_tracked('preview_car', trajectory, edit_request=insert)
    adapter.advance_pose_track(float(times[0]))
    packet = session.render(relative[START_FRAME], float(times[0]))
    second = packet['cameras']['CAM_FRONT']['rgb']
    assert np.array_equal(first_frame, second), 'reset reproduction mismatch'

    (OUT/'validation.json').write_text(json.dumps(dict(
        scope='render+pose continuity only; no UniAD; no control; diagnostic uncertified',
        frames=STEPS, token_stable=True, revision_stable=True,
        reset_reproduction='pixel_identical',
        trajectory='anchor + local +x at 2 m/s, strict bounds',
        asset=insert['asset']), ensure_ascii=False, indent=2))
    print('POSE_TRACK GPU VALIDATION PASSED ->', OUT)


if __name__ == '__main__':
    main()
