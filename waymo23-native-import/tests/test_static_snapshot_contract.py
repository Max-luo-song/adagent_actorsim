import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import ObstacleMemory,PlanningGuard
from runtime_configuration import load_vehicle_config
class StaticSnapshotTests(unittest.TestCase):
 def setUp(self):
  self.m=ObstacleMemory();self.info=dict(scene_token='s',timestamp=0.,declared_static_epoch='generation:revision:time',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3));self.b=np.array([[0.,8.,0.,2.,4.,1.5,0.,0.,8.]])
  self.mask=np.zeros((200,200));self.mask[112:120,98:102]=1
 def update(self,t,visible=True,mask=True):
  self.info['timestamp']=t;rows=[dict(candidate_index=0,raw_score=.9,state='network_retained',label=0,box=self.b[0].tolist())] if visible else [];original=self.b.copy();r=self.m.update(dict(config={},obstacles=rows),self.b,[.9],[0],self.info,current_occupancy=self.mask if mask else None);np.testing.assert_array_equal(self.b,original);return r
 def confirm(self,mask=True):
  for t in [0.,.1,.2]:r=self.update(t,mask=mask)
  return r
 def test_declared_frozen_scene_zeroes_control_motion_not_raw_network_box(self):
  r=self.update(0.);self.assertEqual(r['obstacles'][0]['box'][8],0.);self.assertEqual(r['obstacles'][0]['neural_box'][8],8.)
 def test_validated_shape_does_not_forget_direction_in_immutable_epoch(self):
  self.confirm();self.update(.6,False);r=self.update(1.2,False);g=PlanningGuard(r,load_vehicle_config(),np.zeros((1,5)),[0.]);self.assertEqual(g.rows[0]['geometry'],'static_snapshot_validated_heading_motion_envelope');self.assertTrue(g.rows[0]['heading_support']['static_snapshot']);self.assertTrue(g.rows[0]['hard_constraint'])
 def test_snapshot_revision_change_does_not_inherit_old_validated_heading(self):
  self.confirm();self.info['declared_static_epoch']='different-revision';self.update(.6,False);r=self.update(1.2,False);g=PlanningGuard(r,load_vehicle_config(),np.zeros((1,5)),[0.]);self.assertEqual(g.rows[0]['geometry'],'heading_free_disk_motion_envelope')
 def test_static_metadata_does_not_invent_unobserved_heading(self):
  self.confirm(mask=False);self.update(.6,False);r=self.update(1.2,False);self.assertNotIn('geometry_heading_carry',r['obstacles'][0])
 def test_dynamic_scene_keeps_reported_motion(self):
  del self.info['declared_static_epoch'];r=self.update(0.);self.assertEqual(r['obstacles'][0]['box'][8],8.)
class StaticRasterContractTests(unittest.TestCase):
 def test_declared_snapshot_holds_current_raster_without_mutating_network_forecast(self):
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'waymo172_closed_loop_v1'))
  from closed172_common import optimize
  masks=np.zeros((1,5,200,200));masks[0,0,180,180]=1;masks[0,1:,100,100]=1;original=masks.copy()
  info=dict(model_lidar_frame=dict(virtual_to_ego=np.eye(4)),ego_velo=0.,ego_steer=0.)
  result=optimize(np.zeros((6,2)),masks,info,warm=np.zeros((6,2)),reliability=dict(config={},obstacles=[],declared_static_epoch='g:r:t'))
  self.assertIsNotNone(result['selected'])
  self.assertTrue(result['guard']['declared_static_snapshot'])
  self.assertEqual(result['guard']['occupancy']['occupied_cells'],[1,1,1,1,1])
  np.testing.assert_array_equal(masks,original)

if __name__=='__main__':unittest.main()
