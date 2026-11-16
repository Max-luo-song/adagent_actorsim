
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import unittest,copy,json,pickle,numpy as np
from speed_policy import PlanningGuard
from evidence_guard import PlanningGuard as Before
from runtime_configuration import load_vehicle_config
class SpeedPolicyTests(unittest.TestCase):
 def test_geometry_identical_for_every_evidence_tier(self):
  v=load_vehicle_config();states=np.array([[0.,0.,0.,1.,0.],[0.,.1,0.,1.,0.]])
  for kind,score,n,seconds in [('provisional_uncertain',.11,1,0.),('network_retained',.8,1,0.),('persistent_uncertain',.3,3,.2),('last_known_confirmed',.3,3,.2)]:
   rel=dict(detection_threshold=.4,config={},speed_limit_mps=2.,obstacles=[dict(candidate_index=0,state=kind,raw_score=score,label=0,box=[3.,8.,-1.,2.,4.,1.5,0.,0.,0.],observations=n,duration_seconds=seconds)])
   old=Before(rel,v,states,np.array([0.,.1]));new=PlanningGuard(rel,v,states,np.array([0.,.1]))
   for name in ('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges','soft_boxes','soft_margins'):np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
   np.testing.assert_array_equal(old.clearance(states,[0.,.1]),new.clearance(states,[0.,.1]));self.assertIsNone(new.speed_limit);self.assertEqual(new.caution_speed,v.max_speed_mps)
 def test_report_and_vehicle_are_not_mutated(self):
  v=load_vehicle_config();rel=dict(config={},obstacles=[],speed_limit_mps=2.);before=copy.deepcopy(rel);guard=PlanningGuard(rel,v,np.zeros((2,5)),np.array([0.,.1]));self.assertEqual(rel,before);self.assertEqual(v.max_speed_mps,30.);self.assertEqual(v.max_acc_mps2,2.);self.assertTrue(guard.audit()['speed_policy']['strategy_cap_disabled'])
 def test_live_driver_has_no_preview_task_constraints(self):
  from run_driver import configure
  configure();self.assertEqual(base.boundary_optimize.__module__,'empty_controller');self.assertIs(base.boundary_optimize.__globals__['PlanningGuard'],PlanningGuard)
  self.assertNotIn('COMPLETION_TASK',base.boundary_optimize.__globals__)
  req=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()));self.assertIsNone(req['plan_patch']);self.assertFalse(req['termination']['stall_check']);self.assertFalse(req['termination']['pass_check'])
if __name__=='__main__':unittest.main(verbosity=2)
