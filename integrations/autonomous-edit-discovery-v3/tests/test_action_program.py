import copy,unittest
import numpy as np
from action_program import Timeline,validate_phases

def phase(action,duration,speed,lane='keep'):
 return dict(action=action,duration_s=duration,target_speed_ratio=speed,lane=lane)
def sequence():return [phase('lane_change',3.2,1.,'ego_lane'),phase('decelerate',3.,.3),phase('accelerate',3.,1.)]

class ActionProgramTests(unittest.TestCase):
 def test_merge_brake_accelerate_has_real_velocity_recovery(self):
  t=Timeline(sequence());clock=np.arange(0.,18.8,.001);source,w=t.evaluate(clock);speed=np.gradient(source,.001)
  self.assertLess(speed[np.argmin(abs(clock-7.7))],.301)
  self.assertGreater(speed[np.argmin(abs(clock-11.))],.999)
  self.assertTrue(np.all(np.diff(source)>=-1e-10));self.assertGreater(w[-1],.999)
 def test_stop_hold_and_restart_are_real_not_labels(self):
  t=Timeline([phase('decelerate',3.,0.),phase('stop_hold',1.5,0.),phase('accelerate',3.,.8)])
  src,_=t.evaluate([4.5,5.,6.,9.]);self.assertAlmostEqual(src[0],src[1]);self.assertAlmostEqual(src[0],src[2]);self.assertGreater(src[3],src[2])
 def test_lane_return_is_continuous_and_does_not_teleport(self):
  t=Timeline(sequence()+[phase('lane_change',3.2,1.,'original_lane')])
  _,w=t.evaluate([1.5,3.1,4.7,10.7,12.3,13.9]);np.testing.assert_allclose(w,[0.,.5,1.,1.,.5,0.],atol=1e-12)
 def test_phase_boundaries_have_continuous_position_speed_and_acceleration(self):
  t=Timeline(sequence());h=1e-4
  for end in [1.5,4.7,7.7,10.7]:
   src,_=t.evaluate(end+np.arange(-2,3)*h);v=np.diff(src)/h
   self.assertLess(np.max(abs(np.diff(v))),1e-5)
 def test_source_clock_is_exact_before_activation(self):
  c=np.arange(0,1.5001,.1);s,w=Timeline(sequence()).evaluate(c);np.testing.assert_array_equal(s,c);np.testing.assert_array_equal(w,np.zeros_like(w))
 def test_integral_matches_numerical_speed_profile(self):
  t=Timeline(sequence());s,_=t.evaluate([1.5,4.7,7.7,10.7]);np.testing.assert_allclose(np.diff(s),[3.2,1.95,1.95],atol=1e-12)
 def test_action_labels_cannot_hide_opposite_speed_or_illegal_hold(self):
  bad=[[phase('accelerate',3.,.3),phase('decelerate',3.,0.)],[phase('stop_hold',3.,0.),phase('accelerate',3.,1.)],[phase('cruise',3.,.3),phase('accelerate',3.,1.)]]
  for seq in bad:
   with self.assertRaises(ValueError):validate_phases(seq)
 def test_simple_displacements_and_noop_phases_are_rejected(self):
  with self.assertRaises(ValueError):validate_phases([phase('lane_change',3.,1.,'ego_lane'),phase('cruise',3.,1.)])
  with self.assertRaises(ValueError):validate_phases([phase('cruise',3.,1.),phase('cruise',3.,1.)])
 def test_time_window_and_nonfinite_and_fast_lateral_rejected(self):
  for seq in [sequence()*3,[phase('decelerate',float('nan'),.3),phase('accelerate',3.,1.)],[phase('lane_change',1.,1.,'ego_lane'),phase('decelerate',3.,.3)]]:
   with self.assertRaises(ValueError):validate_phases(seq)
 def test_old_parameter_only_backend_cannot_accept_new_action_program(self):
  from scene_designer import validate,PROGRAM
  import legacy_parameter_designer as old
  from core import digest,AgentError
  context={};v=dict(schema='scene_action_proposals_v2',context_sha256=digest(context),decision='propose',note='',edits=[dict(scene=551,program=PROGRAM,parameters=dict(phases=sequence()),hypothesis='new action ordering',expected_legal_response='yield or choose legal corridor')])
  with self.assertRaises(AgentError):old.validate(v,digest(context),[])
  self.assertEqual(len(validate(v,digest(context),[])),1)
