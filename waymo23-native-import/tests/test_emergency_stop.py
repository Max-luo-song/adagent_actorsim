"""Emergency actions respect dynamics bounds without pretending MPC success."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"waymo172_closed_loop_v1"))
from closed172_common import emergency_stop_action, integrate_state


class EmergencyStopTests(unittest.TestCase):
    def test_recorded_172_failure_speed_stops_without_reversing(self):
        state = np.array([0., 0., 0., 1.9146004671935606, .32109534604528944])
        for index in range(10):
            action = emergency_stop_action(state[3])
            self.assertGreaterEqual(action['acc'], -2.)
            self.assertLessEqual(action['acc'], 0.)
            self.assertEqual(action['steer_rate'], 0.)
            state = integrate_state(state, [action['acc'], action['steer_rate']])
            self.assertGreaterEqual(state[3], -1e-12)
        self.assertAlmostEqual(state[3], 0.)
        self.assertLess(np.linalg.norm(state[:2]), 1.)

    def test_stationary_state_holds_and_invalid_inputs_rejected(self):
        self.assertEqual(emergency_stop_action(0.), dict(acc=0., steer_rate=0.))
        for speed in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                emergency_stop_action(speed)
