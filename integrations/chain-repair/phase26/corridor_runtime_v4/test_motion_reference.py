from bootstrap import *
import copy,types,unittest,numpy as np
from motion_memory import ObstacleMemory
from planning_guard import ObstacleMemory as OldMemory
from spatial_reference import CruisePreference,spatial_error
from feasibility_restore import restore
from runtime_configuration import load_vehicle_config
from native_objective import NativeAlignedObjective
class MotionReferenceTests(unittest.TestCase):
 def info(self,t,speed=8.,scene='generic'):
  return dict(timestamp=t,scene_token=scene,simulation_dt=.1,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),ego_velo=speed,ego_steer=0.,model_lidar_frame={'virtual_to_ego':np.eye(4)})
 def packet(self,t,velocity=30.,state='network_retained'):
  box=np.array([[0.,10.+velocity*t,0.,2.,4.,1.5,0.,0.,velocity]]);report=dict(config=dict(association_distance_m=2.,confirmation_observations=3,confirmation_seconds=.2,max_gap_seconds=.75,obstacle_margin_m=.5),detection_threshold=.4,obstacles=[dict(candidate_index=0,raw_score=.8,label=0,state=state,box=box[0].tolist(),observations=1,duration_seconds=0.)]);return report,box,np.array([.8]),np.array([0]),self.info(t)
 def test_fast_vehicle_association_accumulates_evidence(self):
  new=ObstacleMemory();old=OldMemory()
  for t in [0.,.1,.2]:
   report,boxes,scores,labels,info=self.packet(t);a=new.update(report,boxes,scores,labels,info);b=old.update(report,boxes,scores,labels,info)
  self.assertEqual(a['obstacle_memory']['confirmed_tracks'],1);self.assertEqual(b['obstacle_memory']['confirmed_tracks'],0);self.assertTrue(a['obstacles'][0]['geometry_heading_confirmed'])
 def test_one_observation_and_weak_updates_cannot_confirm(self):
  memory=ObstacleMemory();memory.update(*self.packet(0.))
  for t in [.1,.2]:result=memory.update(*self.packet(t,state='provisional_uncertain'))
  self.assertEqual(result['obstacle_memory']['confirmed_tracks'],0)
 def test_static_epoch_and_scene_reset_preserved(self):
  memory=ObstacleMemory()
  for t in [0.,.1,.2]:
   args=list(self.packet(t,velocity=0.));args[-1]['declared_static_epoch']='fixed';result=memory.update(*args)
  self.assertEqual(result['obstacle_memory']['confirmed_tracks'],1);self.assertEqual(memory.tracks[0]['velocity'].tolist(),[0.,0.,0.])
  args=list(self.packet(.3));args[-1]['scene_token']='other';result=memory.update(*args);self.assertTrue(result['obstacle_memory']['reset']);self.assertEqual(result['obstacle_memory']['confirmed_tracks'],0)
 def test_current_geometry_does_not_force_neural_time_speed(self):
  plan=np.c_[np.zeros(6),np.arange(1,7)*1.];fast=np.c_[np.zeros(12),np.arange(1,13)*4.];slow=np.c_[np.zeros(12),np.arange(1,13)]
  self.assertEqual(spatial_error(fast,plan,False),0.);self.assertEqual(spatial_error(slow,plan,False),0.);self.assertGreater(spatial_error(fast,plan,True),spatial_error(slow,plan,True))
 def test_current_curved_geometry_still_penalizes_wrong_side(self):
  plan=np.array([[0.,1.],[.1,2.],[.4,3.],[.8,4.],[1.2,5.],[1.6,6.]])
  self.assertLess(spatial_error(plan,plan,False),spatial_error(plan*np.array([-1.,1.]),plan,False))
 def test_soft_cruise_survives_braking_and_resets_by_scene(self):
  pref=CruisePreference();v=load_vehicle_config();plan=np.c_[np.zeros(6),np.arange(1,7)]
  self.assertEqual(pref.update(plan,self.info(0.,8.),{},v),8.);self.assertEqual(pref.update(plan,self.info(.1,5.),{},v),8.);self.assertEqual(pref.update(plan,self.info(.2,3.,'other'),{},v),3.)
 def objective(self,plan):
  v=load_vehicle_config();clear=lambda s:np.full(len(s),100.);obj=NativeAlignedObjective(plan,v,types.SimpleNamespace(has_road=True,peak_violation=lambda *a,**kw:0.,cost=lambda *a:0.),types.SimpleNamespace(clearance=lambda s,t:clear(s)),types.SimpleNamespace(clearance=lambda s,t:clear(s),soft_cost=lambda *a:0.),clear);obj.epistemic=types.SimpleNamespace(cost=lambda *a:0.);obj.cruise_preference=8.;return obj
 def test_clear_road_prefers_cruise_and_explicit_neural_stop_is_preserved(self):
  t=np.arange(61)*.1;plan=np.c_[np.zeros(6),np.arange(1,7)];obj=self.objective(plan);fast=np.zeros((61,5));fast[:,1]=8*t;fast[:,3]=8.;slow=np.zeros((61,5));slow[:,1]=2*t;slow[:,3]=2.;controls=np.zeros((6,2));reference=np.c_[np.zeros(12),np.arange(1,13)]
  self.assertLess(obj.cost(fast,fast[:,:2],controls,reference),obj.cost(slow,slow[:,:2],controls,reference));stop=self.objective(np.zeros((6,2)));self.assertTrue(stop.stop);self.assertGreater(stop.cost(fast,fast[:,:2],controls,np.zeros((12,2))),stop.cost(np.zeros((61,5)),np.zeros((61,2)),controls,np.zeros((12,2))))
 def test_cruise_detour_starts_are_symmetric_and_do_not_accelerate_to_maximum(self):
  from cruise_seeds import propose
  from empty_controller import rollout
  v=load_vehicle_config();guard=types.SimpleNamespace(hard_boxes=np.empty((0,9)),hard_radii=np.array([]),hard_margins=np.array([]))
  for interval in [.5,1.]:
   seeds,audit=propose(np.array([0.,0.,0.,8.,0.]),v,guard,None,6*interval,interval);ends=[]
   for name,controls in seeds:
    self.assertTrue(np.all(controls[:,0]<=0.));self.assertGreater(np.abs(controls[:,1]).max(),0.);self.assertLessEqual(np.abs(controls[:,1]).max(),np.deg2rad(v.max_steer_rate_degrees)+1e-10);states,_=rollout(controls,8.,0.,np.zeros(2),v.dt,v.wheelbase_m,interval);ends.append(states[-1,0])
   self.assertLess(ends[0],0.);self.assertGreater(ends[1],0.);np.testing.assert_allclose(ends[0],-ends[1],rtol=0.,atol=1e-5);np.testing.assert_allclose(ends[2],-ends[3],rtol=0.,atol=1e-5)
 def test_phase_one_only_admits_original_constraints(self):
  trials=[dict(seed='bad',minimum_constraint=-1.,controls=[[0.,0.]])];constraints=lambda x:np.array([x[0]-1.,x[1]+1.]);objective=lambda x:float(np.sum(x*x));best,attempts=restore(trials,constraints,objective,[(-2.,2.),(-2.,2.)],maxiter=60,starts=1);self.assertIsNotNone(best);self.assertGreaterEqual(float(constraints(best['x']).min()),-1e-5)
  best,_=restore(trials,constraints,objective,[(-.5,.5),(-2.,2.)],maxiter=20,starts=1);self.assertIsNone(best)
if __name__=='__main__':unittest.main(verbosity=2)
