from bootstrap import *
from runtime_configuration import load_vehicle_config
import copy,json,unittest,numpy as np
from evidence_guard import PlanningGuard
from mpc_reuse_v1.native_guard import PlanningGuard as Original
from mpc_reuse_v1.native_temporal import TemporalOccupancy
from cases import load

class EvidencePolicyTests(unittest.TestCase):
 def synthetic(self,kind='provisional_uncertain',score=.11,label=0,observations=1,duration=0.):
  vehicle=load_vehicle_config();states=np.array([[0.,0.,0.,1.,0.],[0.,.1,0.,1.,0.]])
  box=[0.,0.,-1.,1.,2.,1.,0.,0.,0.]
  report=dict(detection_threshold=.4,config={},obstacles=[dict(candidate_index=11,state=kind,raw_score=score,label=label,box=box,observations=observations,duration_seconds=duration)])
  return report,vehicle,states,np.array([0.,.1])
 def test_all_unconfirmed_classes_remain_risk_without_hard_promotion(self):
  for label in range(10):
   with self.subTest(label=label):
    args=self.synthetic(label=label);before=copy.deepcopy(args[0]);old=Original(*args);new=PlanningGuard(*args)
    self.assertTrue(old.rows[0]['hard_constraint']);self.assertFalse(new.rows[0]['hard_constraint'])
    self.assertEqual(len(new.soft_boxes),1);self.assertEqual(len(new.hard_boxes),0)
    self.assertEqual(new.speed_limit,new.caution_speed);self.assertGreater(new.soft_cost(args[2],args[3]),0)
    self.assertEqual(args[0],before);np.testing.assert_array_equal(new.soft_boxes[0],old.hard_boxes[0])
 def test_actual_retained_confirmed_and_confirmed_memory_unchanged(self):
  for kind,score,count,duration in [('network_retained',.8,1,0.),('persistent_uncertain',.2,3,.2),('last_known_confirmed',.2,3,.2)]:
   with self.subTest(kind=kind):
    args=self.synthetic(kind,score,observations=count,duration=duration);old=Original(*args);new=PlanningGuard(*args)
    self.assertEqual(old.rows,new.rows)
    for name in ('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges'):
     np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
 def test_nonurgent_low_candidate_not_deleted(self):
  args=list(self.synthetic());args[0]['obstacles'][0]['box'][1]=40.
  old=Original(*args);new=PlanningGuard(*args);self.assertEqual(old.rows,new.rows);np.testing.assert_array_equal(old.soft_boxes,new.soft_boxes)
 def test_observed_physical_surface_still_blocks_ego(self):
  args=self.synthetic();PlanningGuard(*args)
  masks=np.zeros((1,5,200,200));masks[0,:,160,160]=1
  temporal=TemporalOccupancy(masks,args[1],observed_points=np.array([[0.,0.]]))
  self.assertLess(temporal.clearance(args[2][:1],[0.])[0],0.)
 def test_actual_551_step2_removes_only_unconfirmed_hard_rows(self):
  _,report,info,plan,occ,vehicle,states,times=load(2);before=copy.deepcopy(report)
  old=Original(report,vehicle,states,times,current_occupancy=occ.reshape(5,200,200)[0]);new=PlanningGuard(report,vehicle,states,times,current_occupancy=occ.reshape(5,200,200)[0])
  self.assertLess(old.clearance(states[:1],[0.])[0],0.)
  self.assertGreater(new.clearance(states[:1],[0.])[0],0.)
  self.assertIn(11,new.demoted_candidate_indices)
  old_rows={r['candidate_index']:r for r in old.rows}
  for row in new.rows:
   if row['candidate_index'] not in new.demoted_candidate_indices:self.assertEqual(row,old_rows[row['candidate_index']])
  self.assertEqual(report,before)
 def test_audit_makes_no_free_space_claim(self):
  g=PlanningGuard(*self.synthetic());audit=g.audit()['proposal_admission']
  self.assertFalse(audit['free_space_asserted']);self.assertTrue(audit['raw_occupancy_and_observed_depth_unchanged'])

if __name__=='__main__':unittest.main(verbosity=2)
