import unittest
from types import SimpleNamespace

import numpy as np

from joint_safety import JointReference


class TestRolloutTrackingVariantRescueOnly(unittest.TestCase):
    def test_feasible_normal_rollout_is_not_replaced_by_tracking_variant(self):
        vehicle = SimpleNamespace(
            dt=0.1,
            wheelbase_m=2.8,
            max_steer_rate_degrees=15.0,
            max_steer_degrees=35.0,
            min_acc_mps2=-4.0,
            max_acc_mps2=2.0,
            max_speed_mps=12.0,
        )
        controller = SimpleNamespace()
        controller.v = vehicle
        controller.mount = np.eye(4)
        controller._rollout_variant = JointReference._rollout_variant.__get__(controller)
        controller.terminal_stop = lambda states: states
        controller.joint_certificate = lambda states, boxes: {'passed': True}
        controller.uncertainty_cost = lambda states, boxes, tracks: 0.0
        controller._risk_boxes = []
        controller._risk_tracks = []
        controller._speed_reference_states = np.zeros((31, 5))

        y = np.arange(0.0, 30.1, 0.2)
        guide = {'y': y, 'x': 0.2 * y * y}
        motion = SimpleNamespace(speed=5.0, steer=0.0, last_accel=0.0)

        normal = controller._rollout_variant(guide, motion, 8.0, 2.0, False)
        tracking = controller._rollout_variant(guide, motion, 8.0, 2.0, True)

        self.assertGreater(
            tracking[0][-1, 1] - tracking[0][0, 1],
            normal[0][-1, 1] - normal[0][0, 1] + 1.0,
            'fixture must make the tracking variant materially more attractive',
        )
        selected = JointReference.rollout(controller, guide, motion, 8.0, 2.0)
        np.testing.assert_allclose(selected[0], normal[0])
        np.testing.assert_allclose(selected[1], normal[1])


if __name__ == '__main__':
    unittest.main()
