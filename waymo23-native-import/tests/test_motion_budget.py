import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import PlanningGuard,envelope_separation
from runtime_configuration import load_vehicle_config
class BudgetTests(unittest.TestCase):
 def guard(self,box):
  return PlanningGuard(dict(config={},obstacles=[dict(candidate_index=0,state='network_retained',raw_score=.8,box=box)]),load_vehicle_config(),np.zeros((2,5)),np.array([0.,.1]))
 def test_away_motion_does_not_erase_static_endpoint_clearance(self):
  box=[0,5,0,2,4,1,0,0,100]
  g=self.guard(box);gap=g.clearance(np.zeros((2,5)),np.array([0.,.1]))
  expected=5-1.5-np.hypot(2,4)/2-.5
  np.testing.assert_allclose(gap,expected,atol=1e-12)
 def test_fast_crossing_in_first_half_interval_still_blocked(self):
  g=self.guard([-5,0,0,.5,.5,1,0,200,0]);gap=g.clearance(np.zeros((2,5)),[0.,.1])
  self.assertLess(gap[0],0)
 def test_current_collision_stays_blocked(self):
  self.assertLess(self.guard([0,0,0,2,4,1,0,0,0]).clearance(np.zeros((1,5)),[0.])[0],0)
if __name__ == '__main__': unittest.main()
