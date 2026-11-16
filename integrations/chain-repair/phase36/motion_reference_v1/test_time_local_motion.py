from bootstrap import *
import unittest,types,numpy as np
from time_local_motion import qualify,tube_boxes,paired_rectangle_gap,travel_lower,PlanningGuard
from evidence_guard import PlanningGuard as Previous
from planning_guard import rectangle_separation
from runtime_configuration import load_vehicle_config
class TubeTests(unittest.TestCase):
 def certificate(self):
  history=[(0.,np.array([0.,10.,0.])),(.1,np.array([0.,11.5,0.]))];return qualify(history,[np.array([0.,15.,0.])]*2,[True,True],.1,{'obstacle_margin_m':.5,'max_gap_seconds':.75},np.eye(3))
 def test_qualification_rejects_missing_support_static_and_contradiction(self):
  history=[(0.,np.array([0.,10.,0.])),(.1,np.array([0.,11.5,0.]))];velocity=[np.array([0.,15.,0.])]*2;cfg={'obstacle_margin_m':.5,'max_gap_seconds':.75}
  self.assertIsNotNone(self.certificate());self.assertIsNone(qualify(history,velocity,[True,False],.1,cfg,np.eye(3)));self.assertIsNone(qualify(history[:1],velocity[:1],[True],0.,cfg,np.eye(3)));self.assertIsNone(qualify([(0.,history[0][1]),(.1,history[0][1])],velocity,[True,True],.1,cfg,np.eye(3)));self.assertIsNone(qualify(history,[velocity[0],-velocity[1]],[True,True],.1,cfg,np.eye(3)))
 def test_paired_sat_matches_full_numpy_matrix_diagonal(self):
  rng=np.random.default_rng(14);states=rng.normal(size=(70,5));boxes=rng.normal(size=(70,9));boxes[:,3:5]=rng.uniform(.1,12.,(70,2));boxes[:,7:9]=0.;v=load_vehicle_config();np.testing.assert_allclose(paired_rectangle_gap(states,boxes,v.half_width_m,v.half_length_m),np.diag(rectangle_separation(states,boxes,np.zeros(70),v.half_width_m,v.half_length_m)),rtol=0,atol=1e-12)
 def test_braking_lower_bound_contains_continuous_and_discrete_models(self):
  for speed in [1.,4.,12.,20.]:
   dt=.1;time=np.arange(61)*dt;bound=travel_lower(speed,time,10.,dt);velocity=speed;position=0.;actual=[0.]
   for _ in range(60):velocity=max(0.,velocity-10.*dt);position+=velocity*dt;actual.append(position)
   np.testing.assert_allclose(bound,actual,atol=1e-10);t=np.minimum(time,speed/10.);continuous=speed*t-5*t*t;self.assertTrue(np.all(bound<=continuous+1e-10));self.assertTrue(np.all(np.diff(bound)>=-1e-10))
 def test_tube_contains_braking_acceleration_and_lateral_motion(self):
  cert=self.certificate();box=np.array([0.,10.,0.,2.,4.,1.5,0.,0.,15.]);times=np.arange(61)*.1;tubes=tube_boxes(box,cert,times);radius=np.hypot(2.,4.)/2
  for speed in [cert['speed_lower_mps'],cert['speed_upper_mps']]:
   for acceleration in [-10.,0.,10.]:
    longitudinal=[0.];velocity=speed
    for _ in range(60):velocity=max(0.,velocity+acceleration*.1);longitudinal.append(longitudinal[-1]+velocity*.1)
    # Project exact point centers into the tube's longitudinal and lateral axes.
    centers=np.c_[np.zeros(61),10.+np.array(longitudinal)];self.assertTrue(np.all(np.abs(centers[:,1]-tubes[:,1])+radius<=tubes[:,4]/2+1e-8))
  self.assertGreater(tubes[-1,3],box[3]);self.assertGreater(tubes[-1,1]-tubes[-1,4]/2,box[1]-radius)
 def test_oncoming_crossing_and_lateral_acceleration_remain_inside_tube(self):
  # Same policy for rotated motion, speed range and both lateral directions.
  rng=np.random.default_rng(82)
  for angle in [0.,np.pi/2,np.pi,-np.pi/2,.73]:
   direction=np.array([np.sin(angle),np.cos(angle)]);normal=np.array([direction[1],-direction[0]])
   for speed in [3.,8.,20.]:
    cert=self.certificate();cert.update(direction_xy=direction.tolist(),speed_lower_mps=speed-.7,speed_upper_mps=speed+.7)
    times=np.arange(31)*.1;box=np.array([0.,10.,0.,2.,4.,1.5,angle,*(speed*direction)]);tubes=tube_boxes(box,cert,times);radius=np.hypot(2.,4.)/2
    for side_sign in [-1.,1.]:
     forward=cert['speed_lower_mps'];side=side_sign*cert['transverse_velocity_error_mps'];position=box[:2].copy();positions=[position.copy()]
     for _ in times[1:]:
      forward=max(0.,forward+rng.uniform(-10.,10.)*.1);side+=side_sign*10.*.1;position=position+.1*(forward*direction+side*normal);positions.append(position.copy())
     offset=np.asarray(positions)-tubes[:,:2]
     self.assertTrue(np.all(np.abs(offset@direction)+radius<=tubes[:,4]/2+1e-8));self.assertTrue(np.all(np.abs(offset@normal)+radius<=tubes[:,3]/2+1e-8))
     states=np.zeros((len(times),5));states[:,:2]=positions
     self.assertTrue(np.all(paired_rectangle_gap(states,tubes,.8,1.5)<0.))
 def test_far_motion_uncertainty_is_retained_in_full_braking_envelope(self):
  args=self.context(self.certificate());guard=PlanningGuard(*args);times=args[3];states=args[2].copy();cert=self.certificate();tubes=tube_boxes(guard.hard_boxes[0],cert,times,guard.hard_radii[0]);states[:,:2]=tubes[:,:2]
  full=guard.full_clearance(states,times);current=guard.clearance(states,times);self.assertTrue(np.all(full<0.));np.testing.assert_array_equal(current[times<=guard.motion_lease],full[times<=guard.motion_lease]);self.assertTrue(np.all(np.isposinf(current[times>guard.motion_lease])));self.assertGreater(guard.soft_cost(states,times),0.)
 def context(self,cert=None,state='network_retained'):

  v=load_vehicle_config();states=np.zeros((61,5));states[:,1]=np.arange(61)*.1*2;states[:,3]=2.;box=[-3.,7.,0.,2.,4.,1.5,0.,0.,15.];row=dict(candidate_index=0,state=state,raw_score=.8,label=0,box=box,observations=2,duration_seconds=.1)
  if cert:row['geometry_time_local_motion']=cert
  report=dict(config={},detection_threshold=.4,obstacles=[row],global_evidence_receipt={'timestamp':.1});return report,v,states,np.arange(61)*.1
 def test_stationary_unqualified_stale_and_missing_keep_original_geometry(self):
  for mode in ['unqualified','stale','memory','stationary','opposite','static_epoch']:
   cert=self.certificate() if mode!='unqualified' else None;state='last_known_confirmed' if mode=='memory' else 'network_retained'
   if mode=='stale':cert['timestamp_s']=0.
   args=list(self.context(cert,state))
   if mode=='stationary':args[0]['obstacles'][0]['box'][7:]=[0.,0.]
   if mode=='opposite':args[0]['obstacles'][0]['box'][7:]=[0.,-15.]
   if mode=='static_epoch':args[0]['obstacles'][0]['declared_static_epoch']='frozen'
   before=Previous(*args);after=PlanningGuard(*args);self.assertFalse(after.local_motion);np.testing.assert_array_equal(before.clearance(args[2],args[3]),after.clearance(args[2],args[3]));np.testing.assert_array_equal(before.hard_margins,after.hard_margins)
 def test_current_geometry_and_margins_are_unchanged_when_motion_qualifies(self):
  args=self.context(self.certificate());before=Previous(*args);after=PlanningGuard(*args);self.assertEqual(len(after.local_motion),1);np.testing.assert_array_equal(before.hard_boxes,after.hard_boxes);np.testing.assert_array_equal(before.hard_margins,after.hard_margins);self.assertEqual(before.clearance(args[2],args[3])[0],after.clearance(args[2],args[3])[0])
if __name__=='__main__':unittest.main(verbosity=2)
