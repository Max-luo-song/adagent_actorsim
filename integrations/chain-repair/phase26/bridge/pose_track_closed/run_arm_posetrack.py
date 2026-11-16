"""pose_track + tracking-control closed-loop launcher (diagnostic, uncertified).

Same phase11 reviewed harness; three explicit sys.modules substitutions,
all recorded in the run fingerprint:
  closed172_common -> bridge/fast_control/tracking_common   (iLQR tracking, no MPC)
  planning_guard   -> bridge/fast_control/planning_guard_tracking (honest labels)
  scene_api        -> bridge/pose_track_closed/scene_api_posetrack (pose_track advance)

Control path carries NO obstacle avoidance (tracking_passthrough_unverified);
the moving managed asset is advanced by trajectory, and the harness's post-hoc
reference_check still measures against the STATIC insert anchor — recorded as
gap-to-anchor, not gap-to-moving-asset. See README.md.
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
import sys

HERE = Path(__file__).resolve().parent          # bridge/pose_track_closed
BRIDGE = HERE.parent
JOB = HERE.parents[3]
P11 = JOB / 'chain_repair/phase11'
ROOT = JOB.parents[3]
sys.path[:0] = [str(HERE), str(BRIDGE / 'fast_control'), str(BRIDGE / 'pose_track'),
                str(P11), str(ROOT / 'integrations/sr-hugsim'),
                str(ROOT / 'waymo23-native-import'),
                str(ROOT / 'waymo23-native-import/waymo172_closed_loop_v1'),
                str(ROOT / 'code/HUGSIM')]

# Bind phase11 reviewed copies BEFORE any path-prepending shim imports
# (second launch failure lesson: same-named files in waymo23-native-import).
import drivable_area
import road_route_seed
assert Path(drivable_area.__file__).resolve().parent == P11, drivable_area.__file__
assert Path(road_route_seed.__file__).resolve().parent == P11, road_route_seed.__file__

import tracking_common as _tc
import planning_guard_tracking as _pgt
import scene_api_posetrack as _sap
for _m in (_tc, _pgt, _sap):
    assert Path(_m.__file__).resolve().is_relative_to(BRIDGE), _m.__file__
sys.modules['closed172_common'] = _tc
sys.modules['planning_guard'] = _pgt
sys.modules['scene_api'] = _sap

from paired_closed_loop_arm import run

if __name__ == '__main__':
    assert len(sys.argv) == 3
    run(sys.argv[1], sys.argv[2])
