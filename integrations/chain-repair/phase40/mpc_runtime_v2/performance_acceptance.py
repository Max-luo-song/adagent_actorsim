"""Reject unmeasured performance policies before a subsequent diagnostic."""

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

import hashlib,json,math
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent
def validate_document(d):
 if d.get('passed') is not True or d.get('physical_math_unchanged') is not True:raise ValueError('Performance verification incomplete')
 if len(d.get('paired_frames',[]))<2:raise ValueError('Need two fresh paired frames')
 for row in d['paired_frames']:
  if not row['selected'] or not row['independent_passed'] or row['forecast_recovery']:raise ValueError('Invalid optimized plan')
  if row['candidate_seconds']>=row['baseline_seconds']:raise ValueError('No measured runtime improvement')
  if row['objective_increase_fraction']>.01:raise ValueError('Objective regression exceeds existing 1% comparison gate')
 if not all(d.get('held_out_expansion',{}).get(k) is True for k in ('expanded','original_starts_preserved','best_controls_identical','original_iteration_budget_preserved')):raise ValueError('Full-search fallback not verified')
 return True

def validate(root=HERE):
 p=Path(root);d=_hugsim_expand(json.loads((p/'performance_acceptance.json').read_text()));validate_document(d)
 for name,digest in {**d['runtime_sha256'],**d['artifact_sha256']}.items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed performance evidence/runtime: '+name)
 checks=_hugsim_expand(json.loads((p/'runtime_benchmarks/independent_candidate_checks.json').read_text()))
 if not checks.get('passed') or {row['step'] for row in checks['rows']}!={8,9}:raise ValueError('Missing independent benchmark checks')
 for row in checks['rows']:
  if not row['physical_bounds'] or not row['braking_witness']['physical_bounds'] or row['minimum_hard_gap_m']< -1e-5 or row['braking_witness']['gap_m']< -1e-5:raise ValueError('Failed independent physical/geometry check')
 held=_hugsim_expand(json.loads((p/'held_out_benchmarks/independent_candidate_checks.json').read_text()))
 if not held.get('passed') or {row['step'] for row in held['rows']}!={2}:raise ValueError('Missing held-out independent check')
 for name,digest in d.get('source_input_sha256',{}).items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed benchmark input: '+name)
 return True
