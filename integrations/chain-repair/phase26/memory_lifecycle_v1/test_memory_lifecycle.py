from bootstrap import *
import copy,unittest,numpy as np
from memory_visibility import box_free
from motion_memory import ObstacleMemory
from speed_policy import PlanningGuard
from depth_ambiguity import ObservationViability
from runtime_configuration import load_vehicle_config


class MemoryLifecycleTests(unittest.TestCase):
 def info(self,t=0.):
  matrix=np.array([[1.,0.,0.,0.],[0.,0.,-1.,1.],[0.,1.,0.,0.],[0.,0.,0.,1.]])
  return dict(timestamp=t,scene_token='generic',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),ego_velo=2.,ego_steer=0.,model_lidar_frame={'virtual_to_ego':np.eye(4)},depth_encoding='expected_camera_z_v1',cam_params={'cam':dict(l2c=matrix,intrinsic=dict(W=100,H=100,cx=50.,cy=50.,fovx=np.pi/2,fovy=np.pi/2))})
 def observation(self,z=50.,alpha=1.):
  return dict(depth={'cam':np.full((100,100),z)},depth_alpha={'cam':np.full((100,100),alpha)})
 def packet(self,t,detected=True):
  box=np.array([[0.,10.,0.,2.,4.,2.,0.,0.,0.]]) if detected else np.empty((0,9))
  cfg=dict(association_distance_m=2.,confirmation_observations=3,confirmation_seconds=.2,max_gap_seconds=.75,obstacle_margin_m=.5)
  report=dict(config=cfg,detection_threshold=.4,obstacles=[dict(candidate_index=0,raw_score=.8,label=0,state='network_retained',box=box[0].tolist(),observations=1,duration_seconds=0.)] if detected else [])
  return report,box,np.full(len(box),.8),np.zeros(len(box),int),self.info(t)
 def confirmed(self,static=False):
  m=ObstacleMemory()
  for t in [0.,.1,.2]:
   args=list(self.packet(t))
   if static:args[-1]['declared_static_epoch']='fixed'
   m.update(*args)
  self.assertTrue(m.tracks[0]['confirmed']);return m
 def stale(self,static=False):
  m=self.confirmed(static)
  for t in np.arange(.3,1.21,.1):
   args=list(self.packet(float(t),False))
   if static:args[-1]['declared_static_epoch']='fixed'
   result=m.update(*args,observation=self.observation(7.))
  return m,result
 def guard(self,report):
  v=load_vehicle_config();states=np.zeros((61,5));states[:,3]=2.
  return PlanningGuard(report,v,states,np.arange(61)*.1),v
 def test_complete_reliable_far_depth_contradicts_old_box(self):
  self.assertTrue(box_free(self.packet(0)[1][0],self.observation(),self.info())['free'])
 def test_occluding_surface_never_clears_old_box(self):
  self.assertFalse(box_free(self.packet(0)[1][0],self.observation(7.),self.info())['free'])
 def test_one_occluded_or_unreliable_pixel_blocks_whole_box_clear(self):
  for key,value in [('depth',7.),('depth_alpha',0.)]:
   obs=self.observation();obs[key]['cam'][50,50]=value
   self.assertFalse(box_free(self.packet(0)[1][0],obs,self.info())['free'])
 def test_missing_reliability_or_invalid_depth_does_not_clear(self):
  obs=self.observation();obs.pop('depth_alpha');self.assertFalse(box_free(self.packet(0)[1][0],obs,self.info())['free'])
  for z in [0.,np.nan,np.inf]:self.assertFalse(box_free(self.packet(0)[1][0],self.observation(z),self.info())['free'])
 def test_partial_fov_or_behind_camera_never_clears(self):
  for x,y in [(25.,10.),(0.,-10.)]:
   box=self.packet(0)[1][0].copy();box[:2]=x,y
   self.assertFalse(box_free(box,self.observation(),self.info())['free'])
 def test_depth_budget_is_applied_to_far_exit_not_center(self):
  # Far face at camera-Z=12; center depth=10 would be an unsafe test.
  self.assertFalse(box_free(self.packet(0)[1][0],self.observation(12.4),self.info())['free'])
  self.assertTrue(box_free(self.packet(0)[1][0],self.observation(12.6),self.info())['free'])
 def test_three_independent_free_observations_retire_old_anchor(self):
  m=self.confirmed()
  for t in [.3,.4]:
   result=m.update(*self.packet(t,False),observation=self.observation());self.assertEqual(len(m.tracks),1)
  result=m.update(*self.packet(.5,False),observation=self.observation())
  self.assertEqual(len(m.tracks),0);self.assertEqual(len(result['obstacle_memory']['lifecycle']['retired']),1)
 def test_occlusion_interrupts_retirement_confirmation(self):
  m=self.confirmed()
  for t,z in [(.3,50.),(.4,7.),(.5,50.),(.6,50.)]:m.update(*self.packet(t,False),observation=self.observation(z))
  self.assertEqual(len(m.tracks),1)
  m.update(*self.packet(.7,False),observation=self.observation());self.assertEqual(len(m.tracks),0)
 def test_fresh_detection_resets_free_counter_and_stays_hard(self):
  m=self.confirmed();m.update(*self.packet(.3,False),observation=self.observation())
  result=m.update(*self.packet(.4),observation=self.observation());self.assertEqual(m.tracks[0]['free_count'],0)
  guard,_=self.guard(result);self.assertEqual(len(guard.hard_boxes),1);self.assertEqual(guard.expired_memory_count,0)
 def test_unseen_expired_anchor_is_retained_as_risk_not_retired(self):
  m,result=self.stale();self.assertEqual(len(m.tracks),1)
  row=result['obstacles'][0];self.assertTrue(row['geometry_evidence_expired']);self.assertGreater(row['evidence_age_s'],.75)
  self.assertEqual(result['obstacle_memory']['lifecycle']['retired'],[])
 def test_expired_anchor_leaves_horizon_hard_geometry_but_not_backup(self):
  _,report=self.stale();g,v=self.guard(report)
  self.assertEqual(len(g.hard_boxes),0);self.assertEqual(g.expired_memory_count,1)
  states=np.zeros((3,5));states[:,1]=[7.,8.,9.];times=np.arange(3)*.1
  self.assertTrue(np.isinf(g.clearance(states,times)).all());self.assertLess(g.full_clearance(states,times).min(),0)
  self.assertGreater(g.soft_cost(states,times),0);self.assertLessEqual(g.soft_cost(states,times),1.)
 def test_memory_only_uncertainty_forces_action_backup_checks(self):
  _,report=self.stale();g,v=self.guard(report);info=self.info(1.2)
  ambiguity=ObservationViability(report,info,v,motion_guard=g)
  self.assertTrue(ambiguity.active);self.assertTrue(ambiguity.memory_active)
  ambiguity.known_clear=lambda states:g.full_clearance(states,np.arange(len(states))*.1)
  initial=np.array([0.,0.,0.,8.,0.]);self.assertLess(ambiguity.gap(initial,np.array([0.,0.])),0)
 def test_static_snapshot_retains_hard_geometry(self):
  _,report=self.stale(static=True);g,_=self.guard(report)
  self.assertEqual(g.expired_memory_count,0);self.assertEqual(len(g.hard_boxes),1)
 def test_scene_reset_removes_old_lifecycle_state(self):
  m=self.confirmed();m.update(*self.packet(.3,False),observation=self.observation())
  args=list(self.packet(.4,False));args[-1]['scene_token']='other';result=m.update(*args,observation=self.observation())
  self.assertTrue(result['obstacle_memory']['reset']);self.assertEqual(m.tracks,[])
 def test_static_epoch_change_invalidates_old_memory(self):
  m=self.confirmed();args=list(self.packet(.3,False));args[-1]['declared_static_epoch']='new_edit'
  result=m.update(*args);self.assertTrue(result['obstacle_memory']['reset']);self.assertEqual(m.tracks,[])
 def test_live_optimizer_keeps_memory_only_backup_in_candidate_admission(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  from test_live_schedule import LiveScheduleTests
  from run_driver import configure
  import empty_controller
  configure();v=load_vehicle_config()
  for speed,safe in [(2.,True),(8.,False)]:
   info,report,masks=LiveScheduleTests().packet(speed,0.,'lifecycle_'+str(speed))
   report['obstacles']=[dict(candidate_index=-1,state='last_known_confirmed',label=0,raw_score=.8,box=[0.,8.,0.,2.,4.,2.,0.,0.,0.],motion_model_expired=True,geometry_evidence_expired=True,missing_seconds=2.)]
   identity=lambda fun,start,**kwargs:SimpleNamespace(x=np.asarray(start).copy(),success=False,status=9,message='test seed evaluator')
   with patch.object(empty_controller,'minimize',side_effect=identity):
    result=empty_controller.optimize(np.c_[np.zeros(6),np.arange(1,7)*speed*.5],masks,info,reliability=report,vehicle=v,maxiter=1)
   self.assertTrue(result['guard']['depth_ambiguity']['expired_memory_backup_active'])
   self.assertEqual(result['guard']['hard_proposals'],0)
   self.assertEqual(result['selected'] is not None,safe)
 def test_observations_and_raw_candidates_are_not_mutated(self):
  import pickle
  m=self.confirmed();args=self.packet(.3,False);obs=self.observation();before=pickle.dumps((args,obs),protocol=4)
  m.update(*args,observation=obs);self.assertEqual(pickle.dumps((args,obs),protocol=4),before)


if __name__=='__main__':unittest.main(verbosity=2)
