"""scene_api substitute: SceneSession subclass with pose_track advancement.

DIAGNOSTIC, UNCERTIFIED. Design: bridge/pose_track/闭环接入评审方案.md.
Structural edits keep the reviewed SceneSession.edit() path (revision bump,
UniAD history reset semantics intact). Per-step pose advancement happens in
render(): the harness computes the simulation timestamp and calls
session.render(pose, timestamp); this subclass advances registered
PoseTrack objects to that timestamp first (no revision/token change, asserted
by PoseTrackSession), then renders. The harness file itself is untouched.

Trajectory rides inside the insert_request JSON as an optional
'pose_track_trajectory' field (times_s + asset_to_world 4x4 list, strict
bounds), popped before delegating to the reviewed edit(). Requests without
the field behave exactly like the reviewed session.
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

import importlib.util
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent            # bridge/pose_track_closed
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
BRIDGE = HERE.parents[0]                          # bridge/
for _p in (str(ROOT / 'integrations/sr-hugsim'), str(BRIDGE / 'pose_track'),
           str(ROOT / 'waymo23-native-import')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    'scene_api_reviewed', ROOT / 'integrations/sr-hugsim/scene_api.py')
_real = importlib.util.module_from_spec(_spec)
sys.modules['scene_api_reviewed'] = _real
_spec.loader.exec_module(_real)

# Full public re-export (whitelist shims failed twice on 2026-10-29; see
# bridge/fast_control/README.md errata). Only SceneSession is replaced.
for _name in dir(_real):
    if not _name.startswith('_') and _name != 'SceneSession':
        globals()[_name] = getattr(_real, _name)
del _name


class SceneSession(_real.SceneSession):
    """Reviewed SceneSession + pose_track advancement on render."""

    def edit(self, request):
        request = dict(request)
        track = request.pop('pose_track_trajectory', None)
        result = super().edit(request)
        if track is not None:
            from pose_track import PoseTrajectory
            from session_adapter import PoseTrackSession
            from gpu_applier import make_backend_pose_applier
            if self.backend is None:
                raise RuntimeError('pose_track trajectory requires a loaded scene')
            times = np.asarray(track['times_s'], dtype=float)
            poses = np.asarray(track['asset_to_world_4x4'], dtype=float)
            if poses.shape != (len(times), 4, 4):
                raise ValueError('pose_track_trajectory: poses must be (N,4,4) matching times_s')
            trajectory = PoseTrajectory(times, poses, bounds=track.get('bounds', 'strict'))
            if not hasattr(self, '_pose_track_adapter'):
                self._pose_track_adapter = PoseTrackSession(
                    self, make_backend_pose_applier(self.backend))
            self._pose_track_adapter.registry.insert(request['object_id'], trajectory)
        return result

    def render(self, camera_pose, timestamp_s, **kwargs):
        adapter = getattr(self, '_pose_track_adapter', None)
        if adapter is not None and adapter.registry.object_ids():
            adapter.advance_pose_track(float(timestamp_s))
        return super().render(camera_pose, timestamp_s, **kwargs)
