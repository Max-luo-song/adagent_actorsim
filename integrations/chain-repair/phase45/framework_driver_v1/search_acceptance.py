"""Own paired search-migration acceptance, separate from inherited kernel evidence."""

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
import hashlib,json

def validate_document(d):
 if d.get('schema')!='framework_search_migration_acceptance_v3' or d.get('passed') is not True:raise ValueError('Own search migration not accepted')
 if not d.get('physics_and_objective_preserved') or len(d.get('paired_frames',[]))<2:raise ValueError('Missing paired evidence or preservation contract')
 before=after=0.
 for r in d['paired_frames']:
  if not r.get('both_selected') or not r.get('independent_passed') or r.get('forecast_recovery') or not r.get('identical_inputs') or not r.get('baseline_matches_frozen_v2'):raise ValueError('Invalid paired plan')
  if r['objective_increase_fraction']>.01 or r['minimum_constraint']<-1e-5:raise ValueError('Search quality regressed')
  if not r.get('iteration_budget_preserved') or not r.get('lateral_seeds_visited'):raise ValueError('Search budget or coverage changed')
  before+=r['baseline_seconds'];after+=r['candidate_seconds']
 if after>=before:raise ValueError('No aggregate benefit on the paired inputs')
 held=d.get('held_out_expansion',{})
 if held.get('kind')!='archived_scheduling_replay' or not all(held.get(k) for k in ['expanded','original_starts_preserved','original_iteration_budget_preserved','same_archived_best']):raise ValueError('Full-search fallback replay missing')
 if d.get('new_gpu_executed') is not False or d.get('whole_loop_speedup_verified') is not False:raise ValueError('CPU evidence cannot certify whole-loop speedup')
 return True

def validate(root):
 p=Path(root)
 from migration_contract import validate as validate_migration
 validate_migration(p)
 d=_hugsim_expand(json.loads((p/'search_acceptance.json').read_text()));validate_document(d)
 for name,expected in {**d['runtime_sha256'],**d['artifact_sha256']}.items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=expected:raise ValueError('Search evidence or implementation changed: '+name)
 for name,expected in d['input_sha256'].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Benchmark input changed: '+name)
 return True
