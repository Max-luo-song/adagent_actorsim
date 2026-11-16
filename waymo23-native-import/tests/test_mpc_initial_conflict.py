"""Fixed-state overlap must not be mistaken for solver convergence failure."""
from pathlib import Path
import sys
from unittest.mock import patch
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'waymo172_closed_loop_v1')]
from closed172_common import optimize

class InitialConflictTests(unittest.TestCase):
    def test_overlap_returns_same_infeasible_decision_without_solver(self):
        occupancy=np.zeros((1,5,200,200));occupancy[0,:,100,100]=1
        info=dict(model_lidar_frame=dict(virtual_to_ego=np.eye(4)),ego_velo=1.,ego_steer=0.)
        with patch('closed172_common.minimize',side_effect=AssertionError('initial overlap cannot be optimized away')):
            result=optimize(np.zeros((6,2)),occupancy,info)
        self.assertIsNone(result['selected'])
        self.assertEqual(result['reason'],'fixed_initial_occupancy_conflict')
        self.assertLess(result['initial_conflict']['clearance_m'],0)
        self.assertFalse(result['initial_conflict']['introduced_by_uncertainty_guard'])
        self.assertEqual(result['trials'],[])
