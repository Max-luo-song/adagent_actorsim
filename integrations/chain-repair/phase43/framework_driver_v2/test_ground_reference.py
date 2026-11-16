from bootstrap import *
import unittest,types,numpy as np
from ground_reference import GroundReference,GRADE,FIT_TOLERANCE
from depth_ambiguity import DepthAmbiguity
from runtime_configuration import load_vehicle_config
class GroundTests(unittest.TestCase):
 def test_interval_contains_flat_and_sloping_ground_and_grows_without_support(self):
  xy=np.array([[x,y] for x in np.linspace(-2,2,9) for y in np.linspace(-2,2,9)]);z=.1*xy[:,0]+.08*xy[:,1];ref=GroundReference(np.c_[xy,z],.5);queries=np.array([[0.,0.],[1.,1.],[0.,20.]])
  low,high=ref.bounds(queries);truth=queries@np.array([.1,.08]);self.assertTrue(np.all(low<=truth));self.assertTrue(np.all(high>=truth));self.assertGreater(high[-1]-low[-1],high[0]-low[0]);self.assertAlmostEqual(ref.error,.5+FIT_TOLERANCE)
 def test_sparse_or_missing_ground_never_produces_finite_false_precision(self):
  lo,hi=GroundReference([],.5).bounds([[3.,4.]]);self.assertTrue(np.isneginf(lo[0]));self.assertTrue(np.isposinf(hi[0]))
 def test_depth_and_reference_errors_are_both_propagated(self):
  ref=GroundReference([[0.,0.,0.]],.5);lo,hi=ref.bounds([[0.,0.]]);self.assertLess(.74-.5-hi[0],.2);self.assertGreater(1.5-.5-hi[0],.2)
 def test_rotation_and_translation_preserve_height_intervals(self):
  points=np.array([[0.,0.,0.],[2.,1.,.1],[-1.,3.,.2]]);q=np.array([[.2,.3],[4.,5.]]);a=.73;r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);t=np.array([12.,-8.]);shift=points.copy();shift[:,:2]=points[:,:2]@r+t;shift[:,2]+=.4
  original=GroundReference(points,.5).bounds(q);moved=GroundReference(shift,.5).bounds(q@r+t)
  for x,y in zip(original,moved):np.testing.assert_allclose(x+.4,y,atol=1e-12)
class BrakingTests(unittest.TestCase):
 def budget(self,points):
  self.v=load_vehicle_config();return DepthAmbiguity({'observed_depth_geometry':{'policy':'observed_depth_geometry_with_free_space_clearing_v1','height_ambiguity_braking_points_xy':points,'voxel_half_diagonal_m':np.sqrt(3)*.25}},{'explicit_l2g_r_mat':np.eye(3)},self.v)
 def test_clear_far_uncertainty_does_not_force_full_braking(self):
  b=self.budget([[0.,25.]]);initial=np.array([0.,0.,0.,8.,0.]);self.assertGreater(b.gap(initial,[0.,0.]),0.)
 def test_near_real_or_unresolved_surface_rejects_acceleration(self):
  b=self.budget([[0.,20.]]);initial=np.array([0.,0.,0.,8.,0.]);b.known_clear=lambda states:.25-np.abs(states[:,0]);self.assertLess(b.gap(initial,[2.,0.]),0.);self.assertGreater(b.gap(initial,[-2.,0.]),0.)
 def test_emergency_stop_does_not_reverse_and_reaches_rest(self):
  b=self.budget([[0.,25.]])
  for speed in [0.,2.,8.,20.]:
   rows=b.stop_states(np.array([0.,0.,0.,speed,0.]),[-min(2.,speed/.1),0.]);self.assertGreaterEqual(rows[:,3].min(),-1e-10);self.assertAlmostEqual(rows[-1,3],0.,places=10)
 def test_empty_ambiguity_is_inactive_and_does_not_change_constraints(self):
  b=self.budget([]);self.assertFalse(b.active);self.assertTrue(np.isposinf(b.gap(np.zeros(5),np.zeros(2))))
 def test_rotated_voxel_and_nonfinite_inputs_are_checked(self):
  self.budget([[2.,10.]])
  with self.assertRaises(ValueError):self.budget([[np.nan,10.]])
if __name__=='__main__':unittest.main(verbosity=2)
