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
 if admission.get('schema')!='explicit_bounded_closed_loop_diagnostic_v1' or admission.get('user_request')!='做一版闭环验证看看':raise ValueError('Missing explicit diagnostic request')
 if arm.get('steps')!=30 or arm.get('start_frame')!=0 or scope.get('scene')!=551 or admission.get('scene')!=551 or scope.get('controls')!=30 or admission.get('controls')!=30:raise ValueError('Diagnostic scope expanded')
 if scope.get('mode')!='closed_loop_smoke_diagnostic' or scope.get('simulation_seconds')!=3. or admission.get('simulation_seconds')!=3.:raise ValueError('Invalid diagnostic duration')
 if scope.get('wall_cap_seconds')!=7200 or request.get('termination',{}).get('max_wall_seconds')!=7200 or admission.get('wall_cap_seconds')!=7200:raise ValueError('Wall caps disagree')
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
 validate_scope(read('smoke_scope.json'),read('arm.json'),read('request_template.json'),read('diagnostic_admission.json'))
 validation=read('validation.json')
 if validation.get('passed') is not True or not validation.get('required_checks') or not all(validation['required_checks'].values()):raise ValueError('Own road/runtime validation incomplete')
 for name,digest in validation['checked_file_sha256'].items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed validated runtime: '+name)
 return True
