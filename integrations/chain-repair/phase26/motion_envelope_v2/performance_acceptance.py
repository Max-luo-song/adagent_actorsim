"""Acceptance for this isolated road-cost + runtime fork, using its own evidence."""

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

import hashlib,json
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent

def validate_document(d):
 if d.get('schema')!='framework_road_runtime_acceptance_v1' or d.get('passed') is not True or d.get('physical_math_unchanged') is not True or d.get('objective_changed') is not True:raise ValueError('Own verification incomplete')
 if len(d.get('paired_frames',[]))<2:raise ValueError('Need two paired actual inputs')
 improved=False
 for row in d['paired_frames']:
  if not row['selected'] or not row['independent_passed'] or row['forecast_recovery']:raise ValueError('Invalid optimized plan')
  if row['objective_increase_fraction']>.01:raise ValueError('Objective regression exceeds 1% comparison gate')
  improved |= row['candidate_seconds']<row['baseline_seconds']
 if not improved:raise ValueError('No measured runtime improvement')
 if not all(d.get('held_out_expansion',{}).get(k) is True for k in ('expanded','original_starts_preserved','best_controls_identical','original_iteration_budget_preserved')):raise ValueError('Full-search fallback not verified')
 return True

def validate(root=HERE):
 # Inherited search/kernel evidence remains bound to the previous frozen fork.
 # It does not certify the new memory admission policy or a new loop speedup.
 p=Path(root);d=_hugsim_expand(json.loads((p/'performance_acceptance.json').read_text()));validate_document(d)
 reference=Path(d['inherited_reference_root']).resolve()
 if reference!=p.parent/'road_runtime_v1':raise ValueError('Wrong inherited optimizer evidence root')
 original=_hugsim_expand(json.loads((reference/'performance_acceptance.json').read_text()))
 if d['runtime_sha256']!=original['runtime_sha256'] or d['artifact_sha256']!=original['artifact_sha256']:raise ValueError('Inherited evidence was rebound')
 for name,digest in {**d['runtime_sha256'],**d['artifact_sha256']}.items():
  if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed inherited optimizer evidence: '+name)
 for name in d['unchanged_optimizer_components']:
  if (p/name).read_bytes()!=(reference/name).read_bytes():raise ValueError('Optimizer component differs from inherited evidence: '+name)
 return True
