import copy
import unittest
from evidence import classify,repeat_agrees
from comparison import compare_metrics


def summary(hd=.3):
    return dict(status='completed',controls=188,observations=189,any_anomaly=False,error_stage=None,stop_recovery_unresolved=False,anomalies={},native_scores=dict(scores=dict(executed_path=dict(hdscore=dict(lower=hd,upper=hd,value=hd)))))


class EvidenceTests(unittest.TestCase):
    def test_score_improvement_is_sufficient_without_collision(self):
        a,b=summary(.2),summary(.25)
        self.assertTrue(compare_metrics(a,b)['improvement_demonstrated'])
    def test_score_gain_does_not_claim_improvement_after_infrastructure_failure(self):
        a,b=summary(.2),summary(.25);b['status']='failed'
        self.assertFalse(compare_metrics(a,b)['improvement_demonstrated'])
    def test_short_prefix_score_cannot_be_compared_to_full_horizon(self):
        a,b=summary(.2),summary(.25);a.update(status='terminated',controls=100,observations=101)
        r=compare_metrics(a,b);self.assertIsNone(r['hd_gain']);self.assertFalse(r['improvement_demonstrated'])
    def test_same_or_lower_score_is_not_a_benefit(self):
        for v in (.2,.19):self.assertFalse(compare_metrics(summary(.2),summary(v))['improvement_demonstrated'])
    def test_unresolved_stop_is_not_success(self):
        a,b=summary(.2),summary(.25);b['stop_recovery_unresolved']=True
        self.assertFalse(compare_metrics(a,b)['improvement_demonstrated'])
    def test_unknown_collision_is_not_a_physical_failure_claim(self):
        a,b=summary(),summary();b.update(anomalies=dict(environment_collision=True),collision_depends_only_on_unknown_semantics=True)
        self.assertEqual(classify(a,b,True,True)['classification'],'uncertain_geometry')
    def test_legal_yield_score_drop_is_not_called_a_control_bug(self):
        self.assertFalse(classify(summary(.3),summary(.2),True,True)['eligible'])
    def test_runtime_failure_is_not_mined(self):
        a,b=summary(),summary();b.update(status='failed',error_stage='worker_boot')
        self.assertEqual(classify(a,b,True,True)['classification'],'infrastructure_failure')
    def test_joint_action_failure_requires_repeat(self):
        a,b=summary(),summary();b.update(status='terminated',termination_reason='no_jointly_verified_action')
        r=classify(a,b,True,True);self.assertTrue(r['eligible']);self.assertTrue(repeat_agrees(r,r))
    def test_invalid_edit_is_not_admitted(self):
        self.assertEqual(classify(summary(),summary(),False,True)['classification'],'invalid_intervention')

if __name__=='__main__':unittest.main()

from qualification import qualify_task,generalization_verdict

class QualificationTests(unittest.TestCase):
    def inputs(self):
        a,b=summary(.3),summary(.2);b['comparison_mode']='off'
        edit=dict(passed=True,matched_prefix_and_camera_effect=True)
        feasible=dict(same_world=True,legal_dynamics=True,road_and_collision_passed=True,given_to_online_policy=False)
        repeat=dict(matched_world_and_seed=True,same_failure_mechanism=True)
        return a,b,edit,feasible,repeat
    def test_solvable_reproducible_edited_native_challenge(self):
        self.assertTrue(qualify_task(*self.inputs())['ready_for_patch_attempt'])
    def test_unsolvable_edit_is_not_a_patch_task(self):
        args=list(self.inputs());args[3]['road_and_collision_passed']=False
        self.assertFalse(qualify_task(*args)['ready_for_patch_attempt'])
    def test_no_repeat_not_admitted(self):
        args=list(self.inputs());args[4]={}
        self.assertFalse(qualify_task(*args)['ready_for_patch_attempt'])
    def test_patched_bad_run_cannot_replace_native_baseline(self):
        args=list(self.inputs());args[1]['comparison_mode']='on'
        self.assertFalse(qualify_task(*args)['ready_for_patch_attempt'])
    def test_missing_held_out_cannot_claim_generalization(self):
        pair=dict(valid=True,complete_horizons_comparable=True,hd_gain=.02)
        self.assertFalse(generalization_verdict(dict(improvement_demonstrated=True),pair,{},['172'])['registered_generalization_passed'])
    def test_held_out_regression_blocks_generalization(self):
        pair=dict(valid=True,complete_horizons_comparable=True,hd_gain=.02)
        bad=dict(valid=True,complete_horizons_comparable=True,hd_gain=-.01)
        self.assertFalse(generalization_verdict(dict(improvement_demonstrated=True),pair,{'172':bad},['172'])['registered_generalization_passed'])
