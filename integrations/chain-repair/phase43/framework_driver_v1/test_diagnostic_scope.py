
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import copy,json,unittest
from diagnostic_admission import validate_scope
class ScopeTests(unittest.TestCase):
 def data(self):return [_hugsim_expand(json.loads((HERE/name).read_text())) for name in ('smoke_scope.json','arm.json','request_template.json','diagnostic_admission.json')]
 def test_explicit_bounded_diagnostic_is_allowed(self):self.assertTrue(validate_scope(*self.data()))
 def test_control_or_wall_expansion_is_rejected(self):
  for change in ('steps','wall'):
   scope,arm,request,admission=self.data()
   if change=='steps':arm['steps']=31
   else:request['termination']['max_wall_seconds']=18000
   with self.assertRaises(ValueError):validate_scope(scope,arm,request,admission)
 def test_global_claim_or_new_sensor_is_rejected(self):
  for key in ('formal_evaluation_admitted','global_repair_verified','supplemental_sensors_enabled'):
   data=self.data();data[0][key]=True
   with self.assertRaises(ValueError):validate_scope(*data)
 def test_request_does_not_promote_global_acceptance(self):
  readiness=_hugsim_expand(json.loads((HERE/'readiness.json').read_text()));self.assertFalse(readiness['global_repair_verified']);self.assertFalse(readiness['closed_loop_launch_ready']);self.assertEqual(readiness['diagnostic_launch_ready'],readiness.get('performance_candidate_frozen',False))
if __name__=='__main__':unittest.main(verbosity=2)
