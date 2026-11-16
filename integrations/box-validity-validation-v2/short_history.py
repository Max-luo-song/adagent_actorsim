"""Availability guard around the unchanged native evaluator. No future extrapolation."""

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

def insufficient_result(report):
 observations=report['observations'];indices=[r['index'] for r in observations]
 if indices!=list(range(len(indices))):raise ValueError('Noncontiguous actual observations; never fill gaps')
 if len(indices)>=26:return None
 return {'status':'insufficient_executed_horizon','scores':None,'actual_executed_path_score':None,'executed_path_scored':False,'actual_controls':len(report['trace']),'actual_observations':len(indices),'required_observations':26,'reason':'Actual recorded execution is shorter than the unchanged 2.5-second native scoring horizon; no future ego/actors fabricated','termination_reason':report.get('termination_reason'),'controller_failure_preserved':report['status']!='completed','score_formula_changed':False}

def score(scoring,work,name,write):
 out=Path(work)/name;report=_hugsim_expand(json.loads((out/'run_report.json').read_text()));result=insufficient_result(report)
 if result is None:return scoring.score_arm(work,name)
 write(out/'native_eval/native_scores.json',result);return result
