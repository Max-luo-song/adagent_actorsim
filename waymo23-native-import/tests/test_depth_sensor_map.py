import unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from depth_sensor_map import DepthObstacleMemory
from planning_guard import TemporalOccupancy,occupancy_clearance
from runtime_configuration import load_vehicle_config

class GeometryMapTests(unittest.TestCase):
 def setUp(self):
  self.v=load_vehicle_config();self.m=DepthObstacleMemory(self.v);self.info=dict(depth_encoding='expected_camera_z_v1',scene_token='s',timestamp=0.,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
  self.obs={'depth':{'CAM_FRONT':np.ones((10,10))}}
 def update(self,t,points):
  self.info['timestamp']=t
  with patch('depth_sensor_map.ground_plane',return_value=np.zeros(3)),patch('depth_sensor_map.points_from_depth',return_value=np.array(points,dtype=float).reshape(-1,3)),patch.object(self.m,'_free',return_value=np.zeros(len(self.m.cells),bool)):
   return self.m.update(self.obs,self.info)
 def test_filters_ground_and_overhead_but_keeps_body_obstacle(self):
  xy,a=self.update(0,[[0,4,.05],[1,5,.8],[2,6,3.]])
  self.assertEqual(a['cells'],1);self.assertEqual(len(xy),1)
 def test_confirmed_geometry_survives_blindness_and_time_gap(self):
  for t in [0.,.1,.2]:self.update(t,[[1,5,.8]])
  self.info['timestamp']=10.;xy,a=self.m.update({},self.info)
  self.assertEqual(a['cells'],1);self.assertEqual(len(xy),1)
 def test_repeated_free_space_clears_and_single_free_sample_does_not(self):
  for t in [0.,.1,.2]:self.update(t,[[1,5,.8]])
  for t in [.3,.4,.5]:
   self.info['timestamp']=t
   with patch.object(self.m,'_free',return_value=np.ones(len(self.m.cells),bool)):
    _,a=self.m.update({},self.info)
   if t<.5:self.assertEqual(a['cells'],1)
  self.assertEqual(a['cells'],0)
 def test_occlusion_breaks_free_space_confirmation(self):
  for t in [0.,.1,.2]:self.update(t,[[1,5,.8]])
  for t,empty in [(.3,True),(.4,False),(.5,True),(.6,True)]:
   self.info['timestamp']=t
   with patch.object(self.m,'_free',return_value=np.full(len(self.m.cells),empty)):
    _,a=self.m.update({},self.info)
  self.assertEqual(a['cells'],1)
 def test_rigid_pose_update_transforms_observed_world_geometry(self):
  self.update(0,[[1,5,.8]]);self.info.update(timestamp=.1,explicit_l2g_t=np.array([0.,1.,0.]))
  xy,a=self.m.update({},self.info);self.assertAlmostEqual(xy[0,1],4.25)
 def test_scene_switch_does_not_reuse_previous_scene_points(self):
  self.update(0,[[1,5,.8]]);self.info.update(timestamp=.1,scene_token='other');xy,a=self.m.update({},self.info);self.assertEqual(len(xy),0)
 def test_invalid_depth_is_not_free_space(self):
  info=dict(cam_params={'C':dict(l2c=np.eye(4),intrinsic=dict(W=3,H=3,cx=1.,cy=1.,fovx=1.5,fovy=1.5))})
  obs=dict(depth={'C':np.full((3,3),np.nan)})
  self.assertFalse(self.m._free(np.array([[0.,0.,2.]]),obs,info)[0])

class GeometryConstraintTests(unittest.TestCase):
 def test_observed_obstacle_blocks_even_when_neural_current_bin_misses_it(self):
  masks=np.zeros((1,5,200,200));masks[0,4,180,180]=1
  occ=TemporalOccupancy(masks,load_vehicle_config(),observed_points=[[0.,0.]])
  self.assertLess(occ.clearance(np.zeros((1,5)),[0.])[0],0)
 def test_tree_pruning_matches_full_exact_geometry(self):
  rng=np.random.default_rng(12);v=load_vehicle_config();points=rng.uniform(-40,40,(1000,2));states=rng.uniform(-15,15,(10,5));masks=np.zeros((1,5,200,200));masks[0,4,180,180]=1
  occ=TemporalOccupancy(masks,v,observed_points=points,observed_point_radius=.433)
  np.testing.assert_allclose(occ.clearance(states,np.zeros(10)),occupancy_clearance(states,points,v,.433),atol=1e-12)
if __name__=='__main__':unittest.main()
