
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
HERE=Path(__file__).resolve().parent
CR=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path[:0]=[str(CR/'phase29'),str(CR/'phase28'),str(ROOT/'waymo23-native-import')]
from mpc_optimized_closed_v1 import run_harness as base
import protocol_harness as h
base.harness=h
