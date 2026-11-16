import numpy as np


def reference_states(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) < 2:
        raise ValueError("expected at least two (right, forward) points")
    if not np.isfinite(trajectory).all():
        raise ValueError("nonfinite trajectory")

    states = np.zeros((len(trajectory) + 1, 5))
    states[1:, :2] = trajectory[:, [1, 0]]
    heading = 0.0
    for i, delta in enumerate(np.diff(states[:, :2], axis=0), start=1):
        forward_delta, right_delta = delta
        if np.linalg.norm(delta) > 1e-8:
            heading = np.arctan2(right_delta, forward_delta)
        states[i, 2] = heading
    states[:, 2] = np.unwrap(states[:, 2])
    return states


def traj2control_fixed(trajectory, info):
    from sim.ilqr.lqr import plan2control
    initial = np.array(
        [0., 0., 0., info["ego_velo"], info["ego_steer"]], dtype=float
    )
    if not np.isfinite(initial).all():
        raise ValueError("nonfinite vehicle state")
    acc, rate = plan2control(reference_states(trajectory), initial)
    if not np.isfinite([acc, rate]).all():
        raise RuntimeError("nonfinite controller output")
    return float(acc), float(rate)
