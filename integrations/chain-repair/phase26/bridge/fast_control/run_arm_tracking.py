"""Tracking-passthrough arm launcher: same phase11 harness, no MPC.

Module substitution is explicit and recorded (same pattern as
bridge/opt/run_arm_opt.py): sys.modules['closed172_common'] and
sys.modules['planning_guard'] are replaced BEFORE importing the phase11
harness, so paired_closed_loop_arm binds the tracking variants. The
substitutes keep their own __file__ (honest audit); the harness and every
other reviewed file stay unchanged.
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

HERE = Path(__file__).resolve().parent          # bridge/fast_control
JOB = HERE.parents[3]                            # unified-open-closed-172-20261022-v4
P11 = JOB / 'chain_repair/phase11'
ROOT = JOB.parents[3]
sys.path[:0] = [str(HERE), str(P11), str(ROOT / 'integrations/sr-hugsim'),
                str(ROOT / 'waymo23-native-import'),
                str(ROOT / 'waymo23-native-import/waymo172_closed_loop_v1'),
                str(ROOT / 'code/HUGSIM')]

# Preload the phase11 reviewed copies BEFORE importing tracking_common:
# tracking_common prepends code/HUGSIM and waymo23-native-import to sys.path
# for sim.ilqr, and waymo23-native-import contains same-named drivable_area.py
# and road_route_seed.py. Without preloading, the waymo23 copies shadow the
# reviewed phase11 ones (2026-10-29 second launch failure: road_prediction_memory
# could not import filter_road_speckles from the waymo23 drivable_area).
import drivable_area
import road_route_seed
assert Path(drivable_area.__file__).resolve().parent == P11, drivable_area.__file__
assert Path(road_route_seed.__file__).resolve().parent == P11, road_route_seed.__file__

import tracking_common as _tc
assert Path(_tc.__file__).resolve().parent == HERE
import planning_guard_tracking as _pgt
assert Path(_pgt.__file__).resolve().parent == HERE
sys.modules['closed172_common'] = _tc
sys.modules['planning_guard'] = _pgt

from paired_closed_loop_arm import run

if __name__ == '__main__':
    assert len(sys.argv) == 3
    run(sys.argv[1], sys.argv[2])
