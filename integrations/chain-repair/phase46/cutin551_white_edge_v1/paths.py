
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
HERE=Path(__file__).resolve().parent
CR=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
PHASE26=CR/'phase26'
UPSTREAM=PHASE26/'motion_search_v3'
DRIVER=HERE.parent/'framework_driver_v1'
SCENE_BUNDLE=CR/'phase33/scene_data551_v1/runs/scene551_20261106T050218Z'
RAW=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training/551'))
SCORER=ROOT/'code/HUGSIM/sim/utils/score_calculator.py'
