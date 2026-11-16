
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
import tempfile,unittest
from generic_guard import validate

class GenericRepairGuardTests(unittest.TestCase):
 def call(self,body):
  with tempfile.TemporaryDirectory() as d:
   code='class JointReference:\n def rollout(self, raw):\n  return raw\n';p=Path(d)/'joint_safety.py';p.write_text(code)
   proposal={'patch_instruction':{'operations':[{'before':'  return raw','after':body}]}}
   return validate({'code':d},proposal,{'id':'new_task'})
 def test_general_math_change_can_proceed_to_real_tests(self):self.assertTrue(self.call('  return raw * 0.5')['passed'])
 def test_scene_number_branch_rejected(self):
  with self.assertRaises(ValueError):self.call('  return raw if 551 else raw')
 def test_track_specific_literal_rejected(self):
  with self.assertRaises(ValueError):self.call("  return raw if 'native:RigidNodes:3' else raw")
 def test_new_forced_direction_literal_rejected(self):
  with self.assertRaises(ValueError):self.call("  return raw if 'left' else raw")
 def test_experiment_file_access_rejected(self):
  with self.assertRaises(ValueError):self.call("  open('/tmp/answers'); return raw")
 def test_task_identifier_rejected(self):
  with self.assertRaises(ValueError):self.call("  return raw if 'new_task' else raw")
