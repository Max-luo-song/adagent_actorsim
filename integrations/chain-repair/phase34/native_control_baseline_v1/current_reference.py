"""Current-frame navigation only; the frozen perception history is untouched."""

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
pass  # Path is provided by the portable boundary above
CONFIG=_hugsim_expand(json.loads((Path(__file__).resolve().parent/'reference_policy.json').read_text()))
ROAD_SOFT_COST_WEIGHT=float(CONFIG['road_soft_cost_weight'])
if not 1.<ROAD_SOFT_COST_WEIGHT<1e6:raise ValueError('Invalid road soft-cost weight')
HISTORY_KEYS=('navigation_anchor_reference','navigation_anchor_audit','committed_navigation_reference','committed_path_center_xy','route_commitment_audit','route_reference_policy','spatial_path_tracking')
def prepare_report(report):
 result=dict(report or {})
 for name in HISTORY_KEYS:result.pop(name,None)
 return result
