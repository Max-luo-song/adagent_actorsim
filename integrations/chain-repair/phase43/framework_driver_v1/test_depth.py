from bootstrap import *
import copy,unittest,types,numpy as np
from unittest.mock import patch
from runtime_configuration import load_vehicle_config
from depth_consistency import DepthConsistencyMemory
from global_repair_v2.boundary_policy import EpistemicRisk
from mpc_reuse_v1.native_temporal import TemporalOccupancy

class DepthEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.vehicle=load_vehicle_config();self.memory=DepthConsistencyMemory(self.vehicle)
  self.obs={'depth':{'CAM':np.ones((4,4))}}
  self.info=dict(timestamp=0.,scene_token='synthetic_episode',depth_encoding='expected_camera_z_v1',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
 def update(self,height,now=0.,refresh=True):
  self.info['timestamp']=now
  def observed(memory,obs,info):
   if refresh:
    key=(0,2,int(height//.5));memory.cells[key]=dict(point=np.array([.1,1.1,height]),count=10,seen=now,free_count=0,free_first=now)
   memory.time=now;memory.scene=info['scene_token']
   return np.array([[.25,1.55]]),dict(policy='observed_depth_geometry_with_free_space_clearing_v1',voxel_resolution_m=.5,voxel_half_diagonal_m=np.sqrt(3)*.25,unknown_visibility_points_xy=[],unknown_visibility_geometry=dict(policy='endpoint_supported_unknown_visibility_v2',point_radius_m=np.sqrt(3)*.25,epistemic_only_points_xy=[]))
  with patch('depth_consistency.BoundedDepthMemory.update',observed),patch('depth_consistency.ground_plane',return_value=np.array([0.,0.,0.])),patch('depth_consistency.ground_reference',return_value=types.SimpleNamespace(bounds=lambda xy:(np.full(len(xy),-.62),np.full(len(xy),.62)),audit=lambda:{})):
   return self.memory.update(self.obs,self.info)
 def test_elevated_current_physical_surface_still_hard_blocks(self):
  xy,a=self.update(1.5);self.assertEqual(len(xy),1)
  occ=np.zeros((1,5,200,200));occ[0,:,160,160]=1
  temporal=TemporalOccupancy(occ,self.vehicle,observed_points=xy)
  self.assertLess(temporal.clearance(np.zeros((1,5)),[0.])[0],0.)
 def test_ambiguous_ground_return_is_retained_as_risk(self):
  xy,a=self.update(.4);self.assertEqual(len(xy),0)
  self.assertEqual(len(self.memory.cells),1);self.assertEqual(a['ambiguous_or_expired_cells'],1)
  self.assertFalse(a['unknown_visibility_geometry']['free_space_asserted'])
  risk=EpistemicRisk({'unknown_visibility_geometry':a['unknown_visibility_geometry']},self.info,self.vehicle)
  self.assertTrue(risk.near_path(np.zeros((1,5))));self.assertGreater(risk.cost(np.zeros((1,5))),0.)
 def test_confirmed_count_cannot_renew_stale_geometry(self):
  self.update(1.5);xy,a=self.update(1.5,now=.8,refresh=False)
  self.assertEqual(len(xy),0);self.assertEqual(len(self.memory.cells),1)
  self.assertEqual(a['cell_evidence'][0]['reason'],'observation_lease_expired')
 def test_same_history_replay_without_observation_does_not_renew_lease(self):
  self.update(1.5)
  for now in (.2,.4,.6):xy,a=self.update(1.5,now=now,refresh=False);self.assertEqual(len(xy),1)
  xy,a=self.update(1.5,now=.8,refresh=False);self.assertEqual(len(xy),0)
 def test_fresh_surface_reacquisition_restores_hard_evidence(self):
  self.update(1.5);self.update(1.5,now=.8,refresh=False);xy,a=self.update(1.5,now=1.,refresh=True)
  self.assertEqual(len(xy),1);self.assertEqual(a['cell_evidence'][0]['last_seen_s'],1.)
 def test_small_ambiguous_surface_is_not_erased_or_left_without_braking_protection(self):
  xy,audit=self.update(.8);self.assertEqual(len(xy),0);self.assertEqual(len(self.memory.cells),1);self.assertEqual(len(audit['height_ambiguity_braking_points_xy']),1)
  from depth_ambiguity import DepthAmbiguity
  budget=DepthAmbiguity({'observed_depth_geometry':audit},self.info,self.vehicle);self.assertTrue(budget.active);self.assertLess(budget.gap(np.array([0.,0.,0.,1.,0.]),np.array([2.,0.])),0.)
 def test_expiry_cannot_silently_clear_a_previously_hard_ambiguous_surface(self):
  self.update(.8);xy,audit=self.update(.8,now=.8,refresh=False);self.assertEqual(len(xy),0);self.assertEqual(len(audit['height_ambiguity_braking_points_xy']),1);self.assertEqual(audit['cell_evidence'][0]['last_seen_s'],0.)
 def test_sensor_packet_and_pose_are_not_modified(self):
  before=copy.deepcopy(self.obs);r=self.info['explicit_l2g_r_mat'].copy();self.update(.4)
  np.testing.assert_array_equal(self.obs['depth']['CAM'],before['depth']['CAM']);np.testing.assert_array_equal(self.info['explicit_l2g_r_mat'],r)

if __name__=='__main__':unittest.main(verbosity=2)
