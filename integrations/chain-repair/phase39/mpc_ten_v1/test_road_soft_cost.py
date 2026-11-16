from bootstrap import *
import unittest,types,numpy as np
from road_soft_cost import SoftRoadPenalty
from runtime_configuration import load_vehicle_config
class RoadSoftCostTests(unittest.TestCase):
 def road(self,mask):
  return types.SimpleNamespace(mask=mask,bounds=np.array([-10.,-10.,10.,10.]),dx=.5,dy=.5,clearance=lambda states:np.full(len(states),-1.))
 def test_farther_offroad_increases_cost_with_directional_gradient(self):
  mask=np.zeros((40,40),bool);mask[:,16:24]=True;p=SoftRoadPenalty(self.road(mask),load_vehicle_config())
  costs=[p.distance_cost(np.array([[x,0.,0.,1.,0.]]),1) for x in (0.,3.,5.,7.)]
  self.assertEqual(costs[0],0.);self.assertTrue(all(a<b for a,b in zip(costs,costs[1:])))
  self.assertLess(p.distance_cost(np.array([[4.9,0.,0.,1.,0.]]),1),p.distance_cost(np.array([[5.1,0.,0.,1.,0.]]),1))
 def test_centered_road_is_symmetric_and_penalty_is_continuous(self):
  mask=np.zeros((40,40),bool);mask[:,16:24]=True;p=SoftRoadPenalty(self.road(mask),load_vehicle_config())
  self.assertAlmostEqual(p.distance_cost(np.array([[4.,0.,0.,1.,0.]]),1),p.distance_cost(np.array([[-4.,0.,0.,1.,0.]]),1),places=10)
  a=p.distance_cost(np.array([[4.-1e-6,0.,0.,1.,0.]]),1);b=p.distance_cost(np.array([[4.+1e-6,0.,0.,1.,0.]]),1);self.assertLess(abs(a-b),1e-3)
 def test_unknown_road_does_not_invent_a_direction(self):
  p=SoftRoadPenalty(self.road(np.zeros((40,40),bool)),load_vehicle_config());self.assertEqual(p.distance_cost(np.array([[4.,0.,0.,1.,0.]]),1),0.);self.assertFalse(p.audit()['road_cells_present']);self.assertFalse(p.audit()['hard_constraint'])
 def test_model_mask_and_states_are_not_mutated(self):
  mask=np.ones((40,40),bool);saved=mask.copy();states=np.array([[0.,0.,0.,1.,0.]]);before=states.copy();p=SoftRoadPenalty(self.road(mask),load_vehicle_config());self.assertEqual(p.distance_cost(states,1),0.);np.testing.assert_array_equal(mask,saved);np.testing.assert_array_equal(states,before)
if __name__=='__main__':unittest.main(verbosity=2)
