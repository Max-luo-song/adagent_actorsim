import numpy as np
from test_repair import JointReference, make_controller, make_motion


def test_rollout_variant_selection_penalizes_lateral_acceleration():
    jr = make_controller()
    y = np.arange(0.0, 90.0, 0.2)
    x = 4.0 * np.sin(y / 6.0)
    guide = dict(side=0, y=y, x=x, end_y=80.0, clear_y=None)
    motion = make_motion(speed=6.0)
    jr._speed_reference_states = np.zeros((40, 5))
    jr._risk_boxes = np.zeros((0, 10))
    jr._risk_tracks = []
    jr.joint_certificate = lambda states, boxes: dict(passed=True)
    states, controls = jr.rollout(guide, motion, 6.0, 6.0)
    lateral = states[:, 3] ** 2 * np.tan(states[:, 4]) / jr.v.wheelbase_m
    assert float(np.max(np.abs(lateral))) <= 3.0
