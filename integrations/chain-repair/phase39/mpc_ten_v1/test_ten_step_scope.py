
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
import json,unittest
from unittest.mock import patch
class TenStepTests(unittest.TestCase):
 def test_exact_ten_controls_and_one_second(self):
  arm=_hugsim_expand(json.loads((HERE/'arm.json').read_text()));scope=_hugsim_expand(json.loads((HERE/'smoke_scope.json').read_text()));self.assertEqual(arm['steps'],10);self.assertEqual(scope['controls'],10);self.assertEqual(scope['simulation_seconds'],1.);self.assertEqual(scope['wall_cap_seconds'],3600)
 def test_source_runtime_is_byte_identical_to_frozen_phase38(self):
  original=_hugsim_expand(json.loads((HERE/'evidence/verified_runtime_receipt.json').read_text()))['implementation_sha256']
  import hashlib
  for name,digest in original.items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest,name)
 def test_metadata_wrapper_retains_errors_and_reports_requested_steps(self):
  import ten_step_driver
  def base_configure():
   def original(*args,**kwargs):raise RuntimeError('original control failure')
   h._control_loop=original
  report={}
  with patch.object(ten_step_driver,'configure_runtime',base_configure):
   ten_step_driver.configure()
   with self.assertRaisesRegex(RuntimeError,'original control failure'):h._control_loop({'setup':{'steps':10}},None,None,report)
  self.assertIn('10 live',report['termination_policy'])
if __name__=='__main__':unittest.main(verbosity=2)
