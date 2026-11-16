
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
import copy,json,sys,tempfile,types,unittest
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'bridge/closed_loop_integration')]
from full_scene_contract import validate_scope
from run_edited_closed_loop import read_task,validation_scope,provenance
from stage_results import finish
from unittest.mock import patch
class FullSceneTests(unittest.TestCase):
 def setup_scope(self):
  task=read_task(ROOT/'full_task.json');scope=validation_scope(task);spec=types.SimpleNamespace(frames=199,fps=10.,source={'scene':551});arm={'start_frame':0,'initial_speed_mps':8.,'arm':'edited'};return task,scope,spec,arm
 def test_default_v4_is_short_diagnostic(self):
  task=read_task(ROOT/'task.json');self.assertEqual(task['driver'],'framework_memory_corridor551_geometry_v4');self.assertEqual(validation_scope(task)['steps'],30)
 def test_full_scene_uses_last_valid_source_observation(self):
  args=self.setup_scope();self.assertTrue(validate_scope(*args));self.assertEqual(args[1]['steps']+1,args[2].frames);self.assertEqual(args[1]['steps']/args[2].fps,19.8)
 def test_wrong_source_clock_budget_or_initial_condition_is_rejected(self):
  for mode in ['frames','fps','scene','steps','speed']:
   task,scope,spec,arm=self.setup_scope()
   if mode=='frames':spec.frames=198
   elif mode=='fps':spec.fps=20.
   elif mode=='scene':spec.source['scene']=552
   elif mode=='steps':scope['steps']=100
   else:arm['initial_speed_mps']=10.
   with self.assertRaises(ValueError):validate_scope(task,scope,spec,arm)
 def test_full_scene_cannot_be_silently_enabled_for_an_old_driver(self):
  task=_hugsim_expand(json.loads((ROOT/'full_task.json').read_text()));task['driver']='framework_memory551_geometry_v1'
  with tempfile.TemporaryDirectory() as folder:
   f=Path(folder)/'task.json';f.write_text(json.dumps(task))
   with self.assertRaises(ValueError):read_task(f)
 def test_provenance_excludes_mutable_shared_adapter(self):
  receipt=provenance();self.assertTrue(receipt)
  for path in receipt:self.assertIn('/corridor_scene_v4/',path)
 def stage(self,folder,avoidance=True,receipts=True):
  root=Path(folder);(root/'comparison_frames').mkdir();(root/'comparison_frames/000.png').write_bytes(b'fixture')
  data=dict(status='completed',integration_acceptance_passed=True,trace=[{}],observations=[{},{}],validation_scope={'name':'full_scene_19p8s_v1'},bbox_diagnostic_required=True,proposal_geometry_receipt_enabled=True)
  (root/'run_report.json').write_text(json.dumps(data));(root/'integration_acceptance.json').write_text('{"passed":true}')
  for name in ['asset_reference.json','scene_manifest.json']:(root/name).write_text('{}')
  diagnostic=dict(passed_obstacle=avoidance,returned_to_recorded_route=avoidance,no_reported_proxy_overlap=True,avoidance_proxy_success=avoidance)
  boxes=dict(passed=True,planner_receipt_frames=1 if receipts else 0,processed_frames=1,video='bbox.mp4',proximity_counts={})
  def encode(args,**kwargs):
   if '-c:v' in args:Path(args[-1]).write_bytes(b'video fixture')
  with patch('stage_results.subprocess.run',side_effect=encode),patch('avoidance_diagnostics.analyze',return_value=diagnostic),patch('bbox_diagnostics.analyze',return_value=boxes):return finish(root,encoder='fake')
 def test_full_window_and_video_are_not_enough_without_avoidance(self):
  with tempfile.TemporaryDirectory() as d:
   result=self.stage(d,avoidance=False);self.assertFalse(result['avoidance_validation_passed']);self.assertEqual(result['exit_code'],1)
 def test_success_requires_avoidance_and_bbox_receipts(self):
  with tempfile.TemporaryDirectory() as d:
   result=self.stage(d);self.assertTrue(result['avoidance_validation_passed']);self.assertEqual(result['exit_code'],0)
  with tempfile.TemporaryDirectory() as d:
   result=self.stage(d,receipts=False);self.assertEqual(result['bbox_diagnostic_status'],'failed');self.assertEqual(result['exit_code'],1)
if __name__=='__main__':unittest.main()
