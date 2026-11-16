from bootstrap import *
import unittest
import numpy as np
from time_local_motion import qualify, POLICY

class VelocityIntervalTests(unittest.TestCase):
 def estimate(self,positions,velocities,angle=0.):
  rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
  times=np.arange(len(positions))*.1
  history=[(t,rotation@np.array([0.,y,0.])) for t,y in zip(times,positions)]
  reported=[rotation@np.array([0.,v,0.]) for v in velocities]
  return qualify(history,reported,[True]*len(times),times[-1],{'obstacle_margin_m':.5,'max_gap_seconds':.75},rotation)
 def test_disagreement_is_covered_once_not_twice(self):
  cert=self.estimate([0.,2.],[14.,14.]);self.assertIsNotNone(cert)
  self.assertAlmostEqual(cert['speed_lower_mps'],14.-2/3)
  self.assertAlmostEqual(cert['speed_upper_mps'],20.+2/3)
  self.assertEqual(cert['actor_acceleration_budget_mps2'],10.)
  self.assertFalse(cert['uncertainty_is_certified'])
 def test_low_reported_outlier_survives_median_consensus(self):
  cert=self.estimate([0.,1.4,2.8],[9.,14.,14.]);self.assertIsNotNone(cert)
  self.assertAlmostEqual(cert['speed_lower_mps'],9.-2/3)
 def test_faster_and_slower_position_fit_are_both_preserved(self):
  for speed in [6.,14.,22.]:
   cert=self.estimate([0.,speed*.1,speed*.2],[14.,14.,14.]);sources=np.array(cert['longitudinal_velocity_sources_mps']);pad=cert['velocity_interval_padding_mps']
   self.assertLessEqual(cert['speed_lower_mps'],min(speed,14.))
   self.assertGreaterEqual(cert['speed_upper_mps'],max(speed,14.))
   self.assertTrue(np.all(sources-pad>=cert['speed_lower_mps']-1e-10))
   self.assertTrue(np.all(sources+pad<=cert['speed_upper_mps']+1e-10))
 def test_strong_braking_samples_are_not_filtered_out(self):
  cert=self.estimate([0.,1.5,2.9],[16.,15.,14.]);self.assertIsNotNone(cert)
  self.assertAlmostEqual(cert['speed_lower_mps'],14.-2/3)
  self.assertAlmostEqual(cert['speed_upper_mps'],16.+2/3)
 def test_conflicting_or_zero_forward_samples_do_not_invent_motion(self):
  self.assertIsNone(self.estimate([0.,0.,0.],[14.,14.,14.]))
  self.assertIsNone(self.estimate([0.,1.4,2.8],[-3.,14.,14.]))
 def test_pose_rotation_preserves_bounds(self):
  baseline=self.estimate([0.,1.9,3.8],[14.,14.,15.])
  for angle in [.5,1.7,-2.4]:
   cert=self.estimate([0.,1.9,3.8],[14.,14.,15.],angle)
   np.testing.assert_allclose([cert['speed_lower_mps'],cert['speed_upper_mps']],[baseline['speed_lower_mps'],baseline['speed_upper_mps']],atol=1e-10)
   np.testing.assert_allclose(cert['direction_xy'],[0.,1.],atol=1e-10)
 def test_policy_keeps_full_acceleration_and_braking_check(self):
  self.assertEqual(POLICY['actor_acceleration_budget_mps2'],10.)
  self.assertTrue(POLICY['complete_current_action_braking_envelope_retained'])
  self.assertFalse(POLICY['speed_interval_is_certified'])
if __name__=='__main__':unittest.main(verbosity=2)
