"""Known behavior failures block GPU launch independently of unit-test status."""

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
CRITICAL=('empty_controller.py','native_objective.py','motion_memory.py','motion_observer.py','time_local_motion.py','motion_tube_policy.json','spatial_reference.py','cruise_seeds.py','feasibility_restore.py','run_driver.py','motion_policy.py','depth_consistency.py','config.json','ground_reference.py','depth_ambiguity.py','context_factory.py','mpc_schedule.py','speed_policy.py','reference_speed_policy.json','scheduling_policy.json')

def implementation_receipt(root=HERE):
 return {name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in CRITICAL}

def evaluate(root=HERE):
 root=Path(root);failures=[];receipt=root/'behavior_acceptance.json'
 if not receipt.is_file():return dict(passed=False,failures=['Missing current-code behavior acceptance receipt; CPU unit tests alone do not admit GPU closure.'])
 try:
  data=_hugsim_expand(json.loads(receipt.read_text()))
  if data.get('implementation_sha256')!=implementation_receipt(root):failures.append('Behavior evidence belongs to older code/configuration.')
  if data.get('schema')!='known_failure_acceptance_v1':failures.append('Unknown behavior acceptance schema.')
  for name in ('step9_neighbor_envelope','depth_geometry_origin_review','repeated_braking_sequence','startup_and_failure_reporting','cross_scene_motion_and_observation','necessary_and_unnecessary_braking'):
   row=data.get('checks',{}).get(name,{})
   if row.get('passed') is not True:failures.append('Known failure unresolved: '+name)
   artifacts=row.get('artifacts',{})
   if not artifacts:failures.append('Missing supporting artifacts: '+name)
   for relative,digest in artifacts.items():
    path=(root/relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:failures.append('Missing/changed evidence artifact: '+str(relative))
  summary=root/'behavior_replays/step_009_summary.json'
  if not summary.is_file():failures.append('Missing saved step9 replay result.')
  else:
   step=_hugsim_expand(json.loads(summary.read_text()));gap=step.get('minimum_constraint')
   if step.get('selected') is not True or not isinstance(gap,(int,float)) or not math.isfinite(gap) or gap< -1e-5:failures.append('Saved step9 still has no admitted feasible plan.')
 except (OSError,ValueError,TypeError) as exc:failures.append('Invalid behavior evidence: '+repr(exc))
 return dict(passed=not failures,failures=failures)

if __name__=='__main__':
 result=evaluate();print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['passed'] else 1)
