"""Optimized-code arm launcher: same phase11 harness, optimized clearance module.

Module substitution is explicit and recorded: sys.modules['closed172_common'] is
replaced by the validated optimized copy BEFORE importing the phase11 harness,
so paired_closed_loop_arm's `from closed172_common import ...` binds to the
optimized implementation. The module keeps its own __file__ (honest audit),
and the fingerprint records the actual files used.
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

HERE = Path(__file__).resolve().parent          # bridge/opt
JOB = HERE.parents[3]                            # unified-open-closed-172-20261022-v4
P11 = JOB/'chain_repair/phase11'
ROOT = JOB.parents[3]
sys.path[:0] = [str(HERE), str(P11), str(ROOT/'integrations/sr-hugsim'),
                str(ROOT/'waymo23-native-import'), str(ROOT/'waymo23-native-import/waymo172_closed_loop_v1')]

import closed172_common_opt as _cc_opt
assert _cc_opt.__file__.startswith(str(HERE)), _cc_opt.__file__
sys.modules['closed172_common'] = _cc_opt
import drivable_area  # phase11 reviewed copies (unchanged)
import road_route_seed
import temporal_occupancy_fast as _tof
assert _tof.__file__.startswith(str(HERE)), _tof.__file__

from paired_closed_loop_arm import run

if __name__ == '__main__':
    assert len(sys.argv) == 3
    run(sys.argv[1], sys.argv[2])
