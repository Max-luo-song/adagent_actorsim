
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

import json,subprocess,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from stage_results import finish,schedule_audit

class StageTests(unittest.TestCase):
 def setup_run(self,root):
  (root/'comparison_frames').mkdir();(root/'comparison_frames/000.png').write_bytes(b'double')
  (root/'run_report.json').write_text(json.dumps(dict(status='completed',integration_acceptance_passed=True,trace=[{}],observations=[{},{}])))
  (root/'integration_acceptance.json').write_text('{"passed":true}')
 def test_video_timeout_preserves_successful_driving(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.setup_run(root)
   with patch('stage_results.subprocess.run',side_effect=subprocess.TimeoutExpired('encoder',120)):
    result=finish(root,encoder='fake')
   self.assertTrue(result['driving_acceptance_passed']);self.assertEqual(result['postprocess_status'],'failed');self.assertEqual(result['exit_code'],1)
   self.assertTrue(_hugsim_expand(json.loads((root/'run_report.json').read_text()))['integration_acceptance_passed'])
 def test_completed_video_has_decode_and_bounded_threads(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.setup_run(root);calls=[]
   def invoke(args,**kwargs):
    calls.append((args,kwargs))
    if '-c:v' in args:Path(args[-1]).write_bytes(b'video double')
    return subprocess.CompletedProcess(args,0)
   with patch('stage_results.subprocess.run',side_effect=invoke):result=finish(root,encoder='fake')
   self.assertEqual(result['exit_code'],0);self.assertEqual(len(calls),2)
   self.assertIn('-filter_threads',calls[0][0]);self.assertEqual(calls[0][1]['timeout'],120);self.assertEqual(calls[1][1]['timeout'],60)
 def test_timeout_exit_is_not_hidden_by_successful_video(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.setup_run(root)
   def invoke(args,**kwargs):
    if '-c:v' in args:Path(args[-1]).write_bytes(b'video')
    return subprocess.CompletedProcess(args,0)
   with patch('stage_results.subprocess.run',side_effect=invoke):result=finish(root,simulation_exit=124,encoder='fake')
   self.assertEqual(result['exit_code'],124);self.assertEqual(result['postprocess_status'],'completed')
 def test_long_driving_pass_without_avoidance_returns_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.setup_run(root)
   report=_hugsim_expand(json.loads((root/'run_report.json').read_text()));report['validation_scope']={'name':'avoidance_18s_v1'};(root/'run_report.json').write_text(json.dumps(report))
   for name in ('asset_reference.json','scene_manifest.json'):(root/name).write_text('{}')
   diagnostic=dict(passed_obstacle=False,returned_to_recorded_route=False,no_reported_proxy_overlap=True,avoidance_proxy_success=False)
   def invoke(args,**kwargs):
    if '-c:v' in args:Path(args[-1]).write_bytes(b'video')
    return subprocess.CompletedProcess(args,0)
   with patch('stage_results.subprocess.run',side_effect=invoke),patch('avoidance_diagnostics.analyze',return_value=diagnostic):result=finish(root,encoder='fake')
   self.assertTrue(result['driving_acceptance_passed']);self.assertFalse(result['avoidance_validation_passed']);self.assertEqual(result['exit_code'],1)
 def test_schedule_counts_real_solver_invocations(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for i,value in enumerate([dict(trials=[{},{}],guard={'scheduling':{'mode':'full_replan','trigger':'period_elapsed'}}),dict(selected={'continuation':True},trials=[],guard={'scheduling':{'mode':'checked_continuation','full_current_constraints_checked':True}})]):
    folder=root/'uniad_worker'/f'step_{i:03d}';folder.mkdir(parents=True);(folder/'optimization.json').write_text(json.dumps(value))
   result=schedule_audit(root);self.assertEqual(result['recorded_solver_invocations'],2);self.assertEqual(result['checked_continuation_steps'],1);self.assertEqual(result['full_replan_steps'],1)
 def test_uncertainty_behavior_review_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.setup_run(root);report=_hugsim_expand(json.loads((root/'run_report.json').read_text()))
   report.update(uncertainty_bundle={'enabled':True},trace=[{'action':{'acc':v},'state_after':[0,0,0,8,0]} for v in [-2.,-2.,0.,-2.]])
   (root/'run_report.json').write_text(json.dumps(report))
   with patch('stage_results.subprocess.run',side_effect=subprocess.TimeoutExpired('video',120)):result=finish(root,encoder='fake')
   self.assertTrue(result['uncertainty_bundle_enabled']);self.assertEqual(result['braking_review']['maximum_braking_steps'],3);self.assertEqual(result['braking_review']['longest_maximum_braking_streak'],2)
if __name__=='__main__':unittest.main()
