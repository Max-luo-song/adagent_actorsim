
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

import json,tempfile,unittest,types
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from execution_reporting import finish_run
class ReportingTests(unittest.TestCase):
 def run_case(self,root,code):return finish_run(root,code,script='postprocess.py',cwd=root,env={})
 def test_missing_report_is_skipped_and_never_success(self):
  with tempfile.TemporaryDirectory() as temp,patch('execution_reporting.subprocess.run') as call:
   for code in (0,1):
    result=self.run_case(temp,code);self.assertEqual(result['postprocess_status'],'skipped_missing_report');self.assertIsNone(result['postprocess_exit']);self.assertIsNone(result['summary']);self.assertNotEqual(result['exit_code'],0)
   call.assert_not_called()
 def test_postprocess_zero_without_summary_is_failure(self):
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp)/'closed_loop';run.mkdir();(run/'run_report.json').write_text('{}')
   with patch('execution_reporting.subprocess.run',return_value=types.SimpleNamespace(returncode=0)):result=self.run_case(temp,0)
   self.assertEqual(result['postprocess_status'],'failed_missing_summary');self.assertNotEqual(result['exit_code'],0)
 def test_postprocess_failure_and_timeout_are_explicit(self):
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp)/'closed_loop';run.mkdir();(run/'run_report.json').write_text('{}')
   for kwargs in ({'return_value':types.SimpleNamespace(returncode=2)},{'side_effect':TimeoutError('timeout')}):
    with patch('execution_reporting.subprocess.run',**kwargs):result=self.run_case(temp,0)
    self.assertEqual(result['postprocess_status'],'failed');self.assertNotEqual(result['exit_code'],0)
 def test_success_preserves_simulation_failure(self):
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp)/'closed_loop';(run/'analysis').mkdir(parents=True);(run/'run_report.json').write_text('{}');(run/'analysis/summary.json').write_text('{}')
   with patch('execution_reporting.subprocess.run',return_value=types.SimpleNamespace(returncode=0)):
    for code in (0,1):
     result=self.run_case(temp,code);self.assertEqual(result['exit_code'],code);self.assertEqual(result['postprocess_status'],'completed');self.assertTrue(Path(result['summary']).is_file())
if __name__=='__main__':unittest.main(verbosity=2)
