"""Actual saved MPC problem exercises new callback and checked-proposal paths on CPU."""
import copy
import pickle
import sys
import unittest
import numpy as np
import pipeline as p


class ActualProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver=p.configure_source()
        with (p.SOURCE/'data_inputs/train_search/step_000.pkl').open('rb') as handle:cls.data=pickle.load(handle)

    def test_exact_teacher_controls_admitted_without_solver(self):
        from direct_problem import make_problem,PlanningProblem
        data=self.data;opt=self.driver.optimized_controller
        problem=make_problem(opt,data['reference'],data['occupancy'],data['info'],copy.deepcopy(data['kwargs']))
        self.assertIsInstance(problem,PlanningProblem)
        controls=np.asarray(data['source_result']['selected']['controls'])
        checked=problem.evaluate(controls)
        self.assertIsNotNone(checked)
        kwargs=copy.deepcopy(data['kwargs']);kwargs.update(_continuation=None,_proposal_provider=lambda current:dict(selected=current.evaluate(controls),audit=dict(test=True)))
        result=opt.optimize(data['reference'],data['occupancy'],data['info'],**kwargs)
        self.assertEqual(result['policy'],'direct_distilled_policy_checked_v1')
        self.assertEqual(result['trials'],[])
        self.assertFalse(result['ground_truth_used_by_optimizer'])
        np.testing.assert_array_equal(result['selected']['controls'],controls)
        invalid=controls.copy();invalid[0,1]=100.
        self.assertIsNone(problem.evaluate(invalid))


if __name__=='__main__':unittest.main()
