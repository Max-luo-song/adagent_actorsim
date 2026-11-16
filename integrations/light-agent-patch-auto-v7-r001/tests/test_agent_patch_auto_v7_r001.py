import unittest
import numpy as np
from joint_safety import JointReference


class _Motion:
    speed = 4.0
    steer = 0.0
    last_accel = 0.0


class TestRolloutLateralGuard(unittest.TestCase):
    """The rollout selector must never adopt a variant whose peak lateral
    excursion materially exceeds the geometric variant of the same guide."""

    def _make(self):
        jr = JointReference.__new__(JointReference)
        jr._risk_boxes = np.zeros((0, 9))
        jr._risk_tracks = []
        jr.terminal_stop = lambda states: states
        jr.joint_certificate = lambda states, boxes: {"passed": True}
        jr.uncertainty_cost = lambda states, boxes, tracks: 0.0

        def fake_variant(guide, motion, target, horizon, tracking_curvature=False):
            n = 31
            peak = 3.0 if tracking_curvature else 1.0
            states = np.zeros((n, 5))
            states[:, 0] = np.linspace(0.0, peak, n)
            states[:, 1] = np.linspace(0.0, 30.0, n)
            states[:, 3] = target
            return states, np.zeros((n - 1, 2))

        jr._rollout_variant = fake_variant
        reference, _ = fake_variant(None, None, 4.0, 3.0, True)
        # Make the wide-lateral variant the objective winner so the parent
        # selector adopts it: zero deviation for it, positive for baseline.
        jr._speed_reference_states = reference.copy()
        return jr

    def test_selected_variant_never_materially_increases_peak_lateral(self):
        jr = self._make()
        guide = {"y": np.linspace(0.0, 40.0, 201), "x": np.zeros(201)}
        states, _ = jr.rollout(guide, _Motion(), 4.0, 3.0)
        baseline_states, _ = jr._rollout_variant(guide, _Motion(), 4.0, 3.0, False)
        self.assertLessEqual(
            float(np.max(np.abs(states[:, 0]))),
            float(np.max(np.abs(baseline_states[:, 0]))) + 0.25,
            "rollout adopted a variant with materially larger lateral excursion",
        )


if __name__ == "__main__":
    unittest.main()
