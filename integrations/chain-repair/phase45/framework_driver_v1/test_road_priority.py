from bootstrap import *
import unittest,numpy as np
from road_priority import road_priority_cost,POLICY


class RoadPriorityTests(unittest.TestCase):
 def test_increasing_violation_never_saturates(self):
  values=[road_priority_cost(dict(road_support_available=True,road_loss=x)) for x in (0.,.01,.1,1.,10.,100.)]
  self.assertTrue(np.all(np.diff(values)>0));self.assertGreater(values[-1],1e6)
 def test_missing_predicted_road_does_not_invent_a_direction(self):
  self.assertEqual(road_priority_cost(dict(road_support_available=False,road_loss=1.)),0.)
 def test_zero_violation_is_zero_and_policy_has_explicit_weight(self):
  self.assertEqual(road_priority_cost(dict(road_support_available=True,road_loss=0.)),0.)
  self.assertEqual(POLICY['weight'],1000.)
 def test_quadratic_loss_becomes_stronger_for_large_violation(self):
  small=road_priority_cost(dict(road_support_available=True,road_loss=.1));large=road_priority_cost(dict(road_support_available=True,road_loss=1.))
  self.assertGreater(large,10*small)
 def test_invalid_loss_is_not_silently_ignored(self):
  for x in (-1.,np.nan,np.inf):
   with self.assertRaises(ValueError):road_priority_cost(dict(road_support_available=True,road_loss=x))


if __name__=='__main__':unittest.main(verbosity=2)
