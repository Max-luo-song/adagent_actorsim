
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

import json,tempfile,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
import run_edited_closed_loop as driver
from avoidance_diagnostics import evaluate_rows

class AvoidanceTests(unittest.TestCase):
 def samples(self,end):
  rows=[]
  for i in range(end+1):
   pose=np.eye(4);pose[0,3]=i
   rows.append(dict(timestamp=i*.5,source_frame=i,closed_ego_to_world=pose.tolist(),open_ego_to_world=pose.tolist(),closed_speed_mps=2.,inserted_reference_gap_m=1.))
  return rows
 def evaluate(self,rows):
  return evaluate_rows(rows,np.array([[8,-1],[12,-1],[12,1],[8,1]]),dict(half_width_m=.8,half_length_m=1.5,clearance_margin_m=.2))
 def test_near_side_does_not_count_as_pass(self):self.assertFalse(self.evaluate(self.samples(12))['passed_obstacle'])
 def test_past_entire_body_and_hold(self):
  result=self.evaluate(self.samples(17));self.assertTrue(result['avoidance_proxy_success'])
 def test_collision_never_counts_success(self):
  rows=self.samples(17);rows[10]['inserted_reference_gap_m']=0.;self.assertFalse(self.evaluate(rows)['avoidance_proxy_success'])
 def test_side_pass_without_return_is_not_success(self):
  rows=self.samples(17)
  for row in rows:
   pose=np.array(row['closed_ego_to_world']);pose[1,3]=3.;row['closed_ego_to_world']=pose.tolist()
  result=self.evaluate(rows);self.assertTrue(result['passed_obstacle']);self.assertFalse(result['returned_to_recorded_route'])
 def test_extended_scope_is_fixed_and_declared(self):
  task=dict(schema='edited_hugsim_closed_loop_v1',driver='phase35_event_mpc551_geometry_v1',validation='avoidance_18s_v1',edits=[dict(source_insert=True)])
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'task.json';path.write_text(json.dumps(task));self.assertEqual(driver.validation_scope(driver.read_task(path))['steps'],180)
   task['validation']='arbitrary';path.write_text(json.dumps(task))
   with self.assertRaises(ValueError):driver.read_task(path)
 def test_ten_second_scope_is_diagnostic_and_bounded(self):
  task=dict(schema='edited_hugsim_closed_loop_v1',driver='phase35_event_mpc551_geometry_v1',validation='diagnostic_10s_v1',edits=[dict(source_insert=True)])
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'task.json';path.write_text(json.dumps(task));scope=driver.validation_scope(driver.read_task(path))
   self.assertEqual(scope['steps'],100);self.assertEqual(scope['max_wall_seconds'],14400);self.assertNotEqual(scope['name'],'avoidance_18s_v1')
 def test_phase37_scope_cannot_expand_into_old_long_profiles(self):
  task=dict(schema='edited_hugsim_closed_loop_v1',driver='phase37_uncertainty551_geometry_v1',edits=[dict(source_insert=True)])
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'task.json';path.write_text(json.dumps(task));scope=driver.validation_scope(driver.read_task(path));self.assertEqual(scope['steps'],30);self.assertEqual(scope['max_wall_seconds'],7200)
   for name in ('diagnostic_10s_v1','avoidance_18s_v1'):
    task['validation']=name;path.write_text(json.dumps(task))
    with self.assertRaises(ValueError):driver.read_task(path)
if __name__=='__main__':unittest.main()
