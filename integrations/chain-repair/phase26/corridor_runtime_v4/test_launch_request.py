
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
import unittest,tempfile,json
from unittest.mock import patch
from run_driver import configure
class RequestTests(unittest.TestCase):
 def test_optional_snapshot_and_valid_bounds(self):
  self.assertIsNone(h.validate_snapshot_request({},30))
  self.assertIsNone(h.validate_snapshot_request({'snapshot_after_index':None},30))
  for n in (0,29):self.assertEqual(h.validate_snapshot_request({'snapshot_after_index':n},30),n)
 def test_invalid_snapshot_is_rejected(self):
  for n in (True,False,-1,30,1.0,'1'):
   with self.subTest(value=n),self.assertRaises(ValueError):h.validate_snapshot_request({'snapshot_after_index':n},30)
 def test_real_request_reaches_scene_context_before_gpu_load(self):
  configure();request=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()))
  self.assertIsNone(request['snapshot_after_index'])
  class ReachedScene(RuntimeError):pass
  class CPUScene:
   def __enter__(self):raise ReachedScene('CPU startup boundary reached')
   def __exit__(self,*args):return False
  with tempfile.TemporaryDirectory() as temp:
   request['output']=str(Path(temp)/'closed_loop')
   with patch.object(h,'SceneSession',CPUScene),self.assertRaises(ReachedScene):h.run_record(request)
   report=_hugsim_expand(json.loads((Path(request['output'])/'run_report.json').read_text()))
   self.assertIn('CPU startup boundary reached',report['error']);self.assertEqual(len(report['trace']),0)
 def test_early_setup_failure_is_saved(self):
  request=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()))
  with tempfile.TemporaryDirectory() as temp:
   request['output']=str(Path(temp)/'closed_loop')
   with patch.object(h,'_setup',side_effect=RuntimeError('early setup failure')),self.assertRaisesRegex(RuntimeError,'early setup failure'):h.run_record(request)
   report=_hugsim_expand(json.loads((Path(request['output'])/'run_report.json').read_text()))
   self.assertEqual(report['failure_stage'],'startup_before_first_control');self.assertIn('early setup failure',report['error_traceback']);self.assertEqual(report['trace'],[])
   from postprocess import summarize
   summary=summarize(request['output']);self.assertIn('early setup failure',summary['error']);self.assertEqual(summary['controls'],0)
 def test_existing_output_is_never_overwritten(self):
  request=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()))
  with tempfile.TemporaryDirectory() as temp:
   request['output']=temp;path=Path(temp)/'run_report.json';path.write_text('preserve')
   with self.assertRaises(FileExistsError):h.run_record(request)
   self.assertEqual(path.read_text(),'preserve')
if __name__=='__main__':unittest.main(verbosity=2)
