"""Separate explicit short diagnostic authorization from global acceptance."""

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
def validate_scope(scope,arm,request,admission):
 if admission.get('schema')!='explicit_bounded_closed_loop_diagnostic_v1' or admission.get('user_request')!='效果这么明显吗，跑一版十帧的看看':raise ValueError('Missing explicit diagnostic request')
 if arm.get('steps')!=10 or arm.get('start_frame')!=0 or scope.get('scene')!=551 or admission.get('scene')!=551 or scope.get('controls')!=10 or admission.get('controls')!=10:raise ValueError('Diagnostic scope expanded')
 if scope.get('mode')!='closed_loop_smoke_diagnostic' or scope.get('simulation_seconds')!=1. or admission.get('simulation_seconds')!=1.:raise ValueError('Invalid diagnostic duration')
 if scope.get('wall_cap_seconds')!=3600 or request.get('termination',{}).get('max_wall_seconds')!=3600 or admission.get('wall_cap_seconds')!=3600:raise ValueError('Wall caps disagree')
 for obj in (scope,admission):
  for key in ('native_evaluation_admitted','formal_evaluation_admitted','global_repair_verified'):
   if obj.get(key) is not False:raise ValueError('Diagnostic cannot claim full acceptance')
 if scope.get('supplemental_sensors_enabled') is not False or request.get('snapshot_after_index') is not None or request.get('reliability_patch') is not None or request.get('plan_patch') is not None:raise ValueError('Unexpected experiment patch or sensor')
 if request.get('termination',{}).get('pass_check') is not False or request.get('termination',{}).get('stall_check') is not False:raise ValueError('Unexpected early termination rule')
 return True

def validate_launch(root=HERE):
 from performance_acceptance import validate as validate_performance
 validate_performance(root)
 p=Path(root);read=lambda name:_hugsim_expand(json.loads((p/name).read_text()))
 scope=read('smoke_scope.json');arm=read('arm.json');request=read('request_template.json');admission=read('diagnostic_admission.json');validate_scope(scope,arm,request,admission)
 if Path(request['arm_request']).resolve()!=p/'arm.json' or Path(arm['insert_request']).resolve()!=p/'insert_request.json':raise ValueError('Launcher points to another experiment')
 validation=read('validation.json');required=('cpu_regression','startup_and_wiring','legacy_worker_import','independent_saved_geometry','runtime_unchanged_from_verified_source','bash_syntax','performance_acceptance')
 if validation.get('passed') is not True or validation.get('unit_tests',0)<145 or not all(validation.get('required_checks',{}).get(k) is True for k in required):raise ValueError('Diagnostic CPU preflight incomplete')
 for name,digest in validation['checked_file_sha256'].items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed preflight input: '+name)
 source=read('evidence/verified_runtime_receipt.json')
 for name,digest in source['implementation_sha256'].items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError('Runtime differs from verified source: '+name)
 if not read('evidence/source_independent_checks.json').get('passed'):raise ValueError('Missing independent source checks')
 if hashlib.sha256((p/'empty_controller.py').read_bytes()).hexdigest()!=scope['controller_sha256']:raise ValueError('Controller changed')
 return True
