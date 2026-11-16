"""v4 changes objective, seeds and constraints; no inherited speedup claim."""

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
import ast,hashlib,json
PRESERVED=('time_local_motion.py','motion_tube_policy.json','motion_memory.py','motion_observer.py','expired_memory_guard.py','memory_visibility.py','memory_lifecycle_policy.json','depth_ambiguity.py','temporal_context.py','context_factory.py','speed_policy.py','native_contract.py','native_contract.json','mpc_schedule.py','fast_road.py','exact_queries.py','reference_geometry.py','consensus_search.py','runtime_policy.json','spatial_reference.py','reference_speed_policy.json','worker.py','run_driver.py','protocol_harness.py')
CORE=('empty_controller.py','native_objective.py','road_soft_cost.py','road_priority.py','road_priority_policy.json','cruise_seeds.py','corridor_geometry.py','corridor_policy.json')
def validate(root):
 p=Path(root);ref=p.parent/'motion_search_v3';receipt=_hugsim_expand(json.loads((p/'change_receipt.json').read_text()))
 for name in PRESERVED:
  if (p/name).read_bytes()!=(ref/name).read_bytes():raise ValueError('Preserved evidence/physics changed: '+name)
 def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 for name in ['integrate_state','rollout']:
  if ast.dump(function(p/'empty_controller.py',name),include_attributes=False)!=ast.dump(function(ref/'empty_controller.py',name),include_attributes=False):raise ValueError('Physical integration changed')
 for name in CORE:
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=receipt['core_sha256'][name]:raise ValueError('Unvalidated corridor implementation changed: '+name)
 return True
