"""Planner timing and reachability behavior, independent of scene IDs."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'waymo172_closed_loop_v1')]
from closed172_common import rollout,optimize,integrate_state,bounded_seed
from runtime_configuration import load_vehicle_config
from unittest.mock import patch
from types import SimpleNamespace
class SearchTests(unittest.TestCase):
    def test_control_interval_matches_physical_integration(self):
        controls=np.tile([.1,.01],(6,1))
        states,_=rollout(controls,2.,0.,[0,0],control_interval_s=1.)
        state=np.array([0.,0.,0.,2.,0.])
        for _ in range(60):state=integrate_state(state,controls[0])
        self.assertEqual(len(states),61)
        np.testing.assert_allclose(states[-1],state)
    def test_long_horizon_continues_neural_navigation_without_obstacle_truth(self):
        occ=np.zeros((1,5,200,200));occ[0,:,190,190]=1
        info=dict(ego_velo=2.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        r=optimize(np.column_stack([np.zeros(6),np.arange(1,7)]),occ,info,maxiter=35,control_interval_s=1.)
        self.assertIsNotNone(r['selected']);self.assertEqual(r['planning_horizon_s'],6.)
        self.assertGreater(r['selected']['states'][-1][1],11.)
        self.assertFalse(r['ground_truth_used_by_optimizer'])
    def test_future_mask_at_ego_does_not_poison_current_state(self):
        occ=np.zeros((1,5,200,200));occ[0,0,190,190]=1;occ[0,4,100,100]=1
        info=dict(ego_velo=2.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        r=optimize(np.column_stack([np.zeros(6),np.arange(1,7)]),occ,info,maxiter=35)
        self.assertIsNotNone(r['selected'])
        self.assertGreater(r['selected']['states'][-1][1],6.)

    def test_forecast_recovery_keeps_observed_obstacle_hard(self):
        occ=np.zeros((1,5,200,200));occ[0,0,100,100]=1
        info=dict(ego_velo=0.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        result=optimize(np.zeros((6,2)),occ,info,_forecast_recovery=True,maxiter=20)
        self.assertIsNone(result['selected'])
        self.assertEqual(result['reason'],'fixed_initial_occupancy_conflict')

    def test_forecast_recovery_preserves_prediction_risk_and_caution_speed(self):
        occ=np.zeros((1,5,200,200));occ[0,0,190,190]=1
        occ[0,1:,98:108,94:107]=1
        info=dict(ego_velo=1.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        result=optimize(np.column_stack([np.zeros(6),np.arange(1,7)*1.5]),occ,info,
                        _forecast_recovery=True,maxiter=45)
        self.assertIsNotNone(result['selected'])
        self.assertTrue(result['forecast_recovery'])
        self.assertLessEqual(max(s[3] for s in result['selected']['states']),2.+1e-5)
        self.assertEqual(result['guard']['speed_limit_mps'],2.)
        self.assertIn('future masks risk cost',result['occupancy_constraint_policy'])

    def test_braking_turn_seeds_respect_physical_bounds_at_every_sample(self):
        vehicle=load_vehicle_config()
        for speed in (0.,.3,3.9):
            for interval in (.5,1.):
                seed=bounded_seed(np.tile([-2.,.26],(6,1)),speed,1.3,interval,vehicle)
                states,_=rollout(seed,speed,1.3,[0,0],control_interval_s=interval)
                self.assertGreaterEqual(states[:,3].min(),-1e-10)
                self.assertLessEqual(abs(states[:,4]).max(),np.deg2rad(vehicle.max_steer_degrees)+1e-10)

    def test_feasible_path_is_not_discarded_for_optimizer_iteration_limit(self):
        occ=np.zeros((1,5,200,200));occ[0,:,190,190]=1
        info=dict(ego_velo=1.,ego_steer=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)))
        def failed_optimizer(fun,x0,**kwargs):
            return SimpleNamespace(success=False,x=np.array(x0),message='iteration limit')
        with patch('closed172_common.minimize',failed_optimizer):
            result=optimize(np.column_stack([np.zeros(6),np.arange(1,7)]),occ,info)
        self.assertIsNotNone(result['selected'])
        self.assertFalse(result['selected']['solver_success'])
        self.assertGreaterEqual(result['selected']['minimum_constraint'],-1e-5)

if __name__=='__main__':unittest.main()
