
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

import json,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from short_history import insufficient_result,score
class ShortHistoryTests(unittest.TestCase):
 def report(self,n):return {'status':'terminated','trace':[{}]*n,'observations':[{'index':i} for i in range(n+1)],'termination_reason':'no_jointly_verified_action'}
 def test_actual_172_failure_does_not_request_future_actors(self):
  root=Path(__file__).resolve().parents[3];p=root/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/box_validity_validation_v1/cases/172/v15/A/run_report.json';r=_hugsim_expand(json.loads(p.read_text()));self.assertEqual(len(r['trace']),21)
  result=insufficient_result(r);self.assertFalse(result['executed_path_scored']);self.assertIsNone(result['scores']);self.assertTrue(result['controller_failure_preserved'])
 def test_short_history_never_calls_native_evaluator(self):
  for n in [0,14,21,24]:
   with tempfile.TemporaryDirectory() as d:
    out=Path(d)/'A';out.mkdir();(out/'run_report.json').write_text(json.dumps(self.report(n)));native=Mock();writer=Mock();score(native,Path(d),'A',writer);native.score_arm.assert_not_called();writer.assert_called_once()
 def test_sufficient_history_keeps_identical_evaluator_call(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'A';out.mkdir();(out/'run_report.json').write_text(json.dumps(self.report(25)));native=Mock();writer=Mock();score(native,Path(d),'A',writer);native.score_arm.assert_called_once_with(Path(d),'A');writer.assert_not_called()
 def test_missing_actual_frame_is_not_interpolated(self):
  r=self.report(21);r['observations'].pop(3)
  with self.assertRaises(ValueError):insufficient_result(r)
if __name__=='__main__':unittest.main()
