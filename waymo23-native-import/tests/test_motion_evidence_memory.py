"""General regressions for motion evidence and finite validated-heading leases."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning_guard import ObstacleMemory,PlanningGuard,carried_heading_shape
from runtime_configuration import load_vehicle_config

class MotionEvidenceTests(unittest.TestCase):
    def sequence(self, position_speed, detector_speed):
        memory=ObstacleMemory();original=[]
        for t in (0.,.1,.2):
            box=np.array([[0.,8+position_speed*t,0.,2.,4.,1.5,0.,0.,detector_speed]])
            rows=[dict(candidate_index=0,box=box[0].tolist(),raw_score=.9,state='network_retained',label=0)]
            report=dict(config={},obstacles=rows);original.append(box.copy())
            info=dict(scene_token='s',timestamp=t,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
            result=memory.update(report,box,[.9],[0],info)
            np.testing.assert_array_equal(box,original[-1])
            self.assertNotIn('geometry_motion_evidence',report['obstacles'][0])
        return result
    def test_smooth_position_drift_is_not_promoted_to_fast_motion(self):
        r=self.sequence(8.,1.);ob=r['obstacles'][0]
        self.assertAlmostEqual(ob['box'][8],1.)
        self.assertFalse(ob['geometry_motion_evidence']['velocity_fit_accepted'])
        self.assertGreater(ob['geometry_motion_evidence']['fit_velocity_disagreement_m'],.5)
        self.assertEqual(r['obstacle_memory']['rejected_position_velocity_fits'],1)
    def test_short_window_agreement_does_not_authorize_large_horizon_error(self):
        r=self.sequence(6.,5.);ob=r['obstacles'][0]
        self.assertFalse(ob['geometry_motion_evidence']['velocity_fit_accepted'])
        self.assertLess(ob['geometry_motion_evidence']['observed_window_disagreement_m'],.5)
        self.assertGreater(ob['geometry_motion_evidence']['fit_velocity_disagreement_m'],.5)
        self.assertAlmostEqual(ob['box'][8],5.)
    def test_real_consistently_fast_motion_is_preserved(self):
        r=self.sequence(8.,8.);ob=r['obstacles'][0]
        self.assertAlmostEqual(ob['box'][8],8.)
        self.assertTrue(ob['geometry_motion_evidence']['velocity_fit_accepted'])
    def test_agreement_rule_is_class_independent_and_does_not_clip_speed(self):
        r=self.sequence(16.,16.)
        self.assertAlmostEqual(r['obstacles'][0]['box'][8],16.)

class HeadingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.m=ObstacleMemory();self.mask=np.zeros((200,200));self.mask[112:120,98:102]=1
        self.box=np.array([[0.,8.,0.,2.,4.,1.5,0.,0.,0.]])
    def update(self,t,visible=True,mask=True):
        report=dict(config={},obstacles=[dict(candidate_index=0,box=self.box[0].tolist(),raw_score=.9,state='network_retained',label=0)] if visible else [])
        info=dict(timestamp=t,scene_token='s',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3))
        return self.m.update(report,self.box,[.9],[0],info,current_occupancy=self.mask if mask else None)
    def confirm(self,mask=True):
        for t in (0.,.1,.2):self.update(t,mask=mask)
    def test_brief_loss_preserves_recent_independently_validated_shape(self):
        self.confirm();r=self.update(.3,visible=False);ob=r['obstacles'][0]
        self.assertIn('geometry_heading_carry',ob)
        guard=PlanningGuard(r,load_vehicle_config(),np.zeros((1,5)),[0.])
        self.assertEqual(guard.rows[0]['geometry'],'finite_age_validated_heading_motion_envelope')
        self.assertEqual(guard.rows[0]['effective_margin_m'],.5)
        self.assertTrue(guard.rows[0]['hard_constraint'])
    def test_heading_uncertainty_grows_and_expired_lease_returns_disk(self):
        self.confirm();a=self.update(.3,visible=False)['obstacles'][0];b=self.update(.6,visible=False)['obstacles'][0]
        self.assertGreater(b['geometry_heading_carry']['heading_range_rad'],a['geometry_heading_carry']['heading_range_rad'])
        r=self.update(1.,visible=False);self.assertNotIn('geometry_heading_carry',r['obstacles'][0])
        guard=PlanningGuard(r,load_vehicle_config(),np.zeros((1,5)),[0.])
        self.assertEqual(guard.rows[0]['geometry'],'heading_free_disk_motion_envelope')
        self.assertTrue(guard.rows[0]['hard_constraint'])
    def test_no_independent_validation_never_creates_lease(self):
        self.confirm(mask=False);r=self.update(.3,visible=False)
        self.assertNotIn('geometry_heading_carry',r['obstacles'][0])
    def test_weak_or_missing_observation_does_not_refresh_lease(self):
        self.confirm();self.update(.4,visible=False)
        ob=self.update(.7,visible=False)['obstacles'][0]
        self.assertAlmostEqual(ob['geometry_heading_carry']['age_s'],.5)
    def test_carried_obstacle_still_blocks_actual_overlap(self):
        self.confirm();r=self.update(.3,visible=False)
        g=PlanningGuard(r,load_vehicle_config(),np.array([[0.,8.,0.,0.,0.]]),[0.])
        self.assertLess(g.clearance(np.array([[0.,8.,0.,0.,0.]]),np.array([0.]))[0],0)
    def test_unverified_or_expired_manual_carry_is_rejected(self):
        ob=dict(state='last_known_confirmed',geometry_heading_carry=dict(source='unverified'))
        self.assertIsNone(carried_heading_shape(self.box[0],ob))

if __name__=='__main__':unittest.main()
