"""Path-heading error convention for this isolated MPC version.

Public vectors: (..., [forward, right]); positive yaw/error means right.
Legacy MPC vectors: (..., [right, forward]); adapt explicitly at the boundary.
See docs/conventions.md. Angles are radians in the same coordinate frame.
"""
import numpy as np


def wrap_to_pi(angle):
    """Map finite radians to [-pi, pi); exactly opposite maps to -pi."""
    angle = np.asarray(angle, dtype=float)
    if not np.isfinite(angle).all():
        raise ValueError('Heading angles must be finite radians')
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_heading_error(path_vec, ego_heading):
    """Signed target-minus-ego error; vectors are [forward, right].

    Supports scalar vectors and batches via NumPy broadcasting. Positive
    error requests rightward correction. Reject zero vectors: no heading is
    defined for them. Both inputs must refer to the same coordinate frame.
    """
    vector = np.asarray(path_vec, dtype=float)
    heading = np.asarray(ego_heading, dtype=float)
    if vector.ndim < 1 or vector.shape[-1] != 2:
        raise ValueError('Path vector must have shape (..., 2): [forward, right]')
    if not np.isfinite(vector).all() or not np.isfinite(heading).all():
        raise ValueError('Path vectors and headings must be finite')
    if np.any(np.all(vector == 0., axis=-1)):
        raise ValueError('Zero-length path vector has no heading')
    target = np.arctan2(vector[..., 1], vector[..., 0])
    return wrap_to_pi(target - heading)


def mpc_heading_error(tangent_right_forward, ego_heading):
    """Explicit adapter from native MPC [right, forward] coordinates."""
    tangent = np.asarray(tangent_right_forward, dtype=float)
    if tangent.ndim < 1 or tangent.shape[-1] != 2:
        raise ValueError('MPC tangent must have shape (..., 2): [right, forward]')
    return compute_heading_error(tangent[..., [1, 0]], ego_heading)


def spatial_tracking_cost(states, path, sample_stride, control_interval_s,
                          target_speed, wheelbase):
    """Existing spatial tracking cost with only heading convention corrected.

    Sampling, nearest-segment choice, position/speed weights and heading loss
    (1-cos(error)) retain the frozen implementation's semantics.
    """
    states = np.asarray(states, dtype=float)
    path = np.asarray(path, dtype=float)
    segments = np.diff(path, axis=0)
    if len(segments) == 0 or np.any(np.all(segments == 0., axis=1)):
        raise ValueError('Spatial route needs nonzero segments')
    sampled = states[sample_stride::sample_stride]
    points = sampled[:, :2]
    delta = points[:, None, :] - path[None, :-1, :]
    u = np.clip(np.sum(delta * segments[None], axis=2) /
                np.sum(segments * segments, axis=1)[None], 0., 1.)
    closest = path[None, :-1, :] + u[:, :, None] * segments[None]
    error = np.sum((points[:, None, :] - closest)**2, axis=2)
    idx = np.argmin(error, axis=1)
    heading_error = mpc_heading_error(segments[idx], sampled[:, 2])
    return float(np.sum(error[np.arange(len(points)), idx]
        + control_interval_s**2 * (sampled[:, 3] - target_speed)**2
        + wheelbase**2 * (1. - np.cos(heading_error))))
