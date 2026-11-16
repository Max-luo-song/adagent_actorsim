"""Unsaturated road cost, separate from the bounded metric-alignment score."""

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

import json,math
pass  # Path is provided by the portable boundary above
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name('road_priority_policy.json').read_text()))
if POLICY['schema']!='unbounded_model_road_priority_v1' or not 0<float(POLICY['weight'])<1e6:
    raise ValueError('Invalid road-priority policy')


def road_priority_cost(metrics,policy=None):
    policy=POLICY if policy is None else policy
    if not policy['enabled'] or not metrics['road_support_available']:return 0.
    loss=float(metrics['road_loss'])
    if not math.isfinite(loss) or loss<0:raise ValueError('Invalid road loss')
    peak=float(metrics.get('road_peak_violation_m2',0.))
    if not math.isfinite(peak) or peak<0:raise ValueError('Invalid peak road loss')
    return float(policy['weight'])*loss*(1.+loss)+float(policy.get('peak_weight',0.))*peak
