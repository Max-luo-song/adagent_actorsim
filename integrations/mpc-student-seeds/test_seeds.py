import copy
import pickle
import unittest
from types import SimpleNamespace
import numpy as np
from seed_adapter import candidates,admit,build,SeedProvider
import experiment as e


class ScheduleTests(unittest.TestCase):
    def original(self,search,seeds,budget,evaluate):
        for name in ['warm','coast','escape_left','escape_right']:yield name,np.zeros((6,2)),budget,'consensus_primary'
        if search.agreement():return
        for name in ['old_a','old_b']:yield name,np.zeros((6,2)),budget,'consensus_expansion'
    def test_no_inference_when_original_primary_finishes(self):
        search=SimpleNamespace(agreement=lambda:True);audit={}
        def forbidden():self.fail('Student must not run before original primary gate')
        rows=list(candidates(self.original,search,[],220,None,forbidden,audit))
        self.assertEqual(len(rows),4);self.assertFalse(audit['student_seed']['inference_executed'])
    def test_one_full_budget_extra_then_complete_original_fallback(self):
        search=SimpleNamespace(agreement=lambda:False);audit={}
        rows=list(candidates(self.original,search,[],220,None,lambda:([('student_initial_mode_2',np.ones((6,2)))],dict()),audit))
        self.assertEqual([r[0] for r in rows],['warm','coast','escape_left','escape_right','student_initial_mode_2','old_a','old_b'])
        self.assertTrue(all(r[2]==220 for r in rows))
    def test_only_original_agreement_can_skip_remaining_starts(self):
        calls=[False]
        search=SimpleNamespace(agreement=lambda:calls[0]);audit={}
        iterator=candidates(self.original,search,[],220,None,lambda:([('student_initial_mode_2',np.zeros((6,2)))],dict()),audit)
        for _ in range(5):next(iterator)
        calls[0]=True
        self.assertEqual(list(iterator),[])
        self.assertTrue(audit['student_seed']['remaining_original_starts_skipped_by_original_agreement'])
    def test_unconverged_or_unoptimized_student_cannot_be_selected(self):
        search=SimpleNamespace(acceptable=lambda row:row['feasible']);row=dict(solver_success=True,used_verified_seed=False,optimization_skipped=False,feasible=True)
        self.assertTrue(admit(row,search))
        for key,value in [('solver_success',False),('used_verified_seed',True),('optimization_skipped',True),('feasible',False)]:
            bad=dict(row);bad[key]=value;self.assertFalse(admit(bad,search))


class ActualAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.driver=e.old.configure_source()
    def test_isolated_clone_preserves_source_globals_and_public_optimizer(self):
        original=self.driver.optimized_controller;function=original.optimize
        controller,receipt=build(original,lambda *args:([],{}))
        self.assertIs(original.optimize,function)
        self.assertIs(controller.optimize.__wrapped__.__globals__['PlanningGuard'],original.PlanningGuard)
        self.assertIsNot(controller.optimize,function)
        self.assertFalse(receipt['source_files_modified'])
        self.assertIn('student_post_solve_admitted',receipt['compiled_function_source'])
    def test_real_network_only_returns_one_bounded_initial_guess(self):
        trained=e.read(e.PARENT/'student/receipt.json');provider=SeedProvider(trained['checkpoint']['path'],trained['checkpoint']['sha256'])
        with open(e.SOURCE/'data_inputs/test_cutin/step_000.pkl','rb') as f:data=pickle.load(f)
        from direct_problem import make_problem
        original=self.driver.optimized_controller
        problem=make_problem(original,data['reference'],data['occupancy'],data['info'],copy.deepcopy(data['kwargs']))
        def score(name,seed):return float(problem.constraints(seed.ravel()).min()),float(problem.objective(seed.ravel()))
        rows,receipt=provider(problem,[('coast',np.zeros((6,2)))],score)
        self.assertEqual(len(rows),1);self.assertFalse(receipt['direct_control'])
        bounds=np.asarray(problem.bounds);u=rows[0][1].ravel()
        self.assertTrue(np.all(u>=bounds[:,0]-1e-8) and np.all(u<=bounds[:,1]+1e-8))


if __name__=='__main__':unittest.main()
