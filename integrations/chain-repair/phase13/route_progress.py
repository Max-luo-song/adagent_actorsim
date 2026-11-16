"""Signed, local arclength progress on a geometric route; never a safety exemption."""
import numpy as np


def projected_arclength(points, path):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if not len(segments) or np.any(lengths <= 1e-8) or not np.isfinite(points).all() or not np.isfinite(path).all():
        raise ValueError('Progress requires a finite, nondegenerate path')
    offsets = points[:, None, :] - path[None, :-1, :]
    u = np.clip(np.sum(offsets*segments[None], axis=2) / lengths[None]**2, 0., 1.)
    closest = path[None, :-1, :] + u[:, :, None]*segments[None]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    chosen = np.argmin(distances, axis=1)
    arcs = np.r_[0., np.cumsum(lengths)]
    return arcs[chosen] + u[np.arange(len(points)), chosen]*lengths[chosen], distances[np.arange(len(points)), chosen]


def predicted_window_progress(states, path, dt, prediction_s, target_speed, half_width):
    """Reward forward route progress inside the predicted occupancy window.

    This is a cost term, never a certificate of free space: every candidate still
    passes the full collision, road, vehicle and detection constraints.
    """
    states = np.asarray(states, dtype=float)
    count = min(len(states)-1, int(np.floor(prediction_s/dt + 1e-9)))
    if count < 1 or target_speed <= 0: return 0.
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if not len(lengths) or np.any(lengths <= 1e-8):
        raise ValueError('Progress requires a nondegenerate path')
    midpoints = (states[:count, :2] + states[1:count+1, :2])/2
    offsets = midpoints[:, None, :] - path[None, :-1, :]
    u = np.clip(np.sum(offsets*segments[None], axis=2)/lengths[None]**2, 0., 1.)
    closest = path[None, :-1, :] + u[:, :, None]*segments[None]
    dist = np.linalg.norm(midpoints[:, None, :] - closest, axis=2)
    # Soft nearest-segment tangents avoid a discontinuous gradient when a
    # candidate crosses a corner or a rasterized route vertex.
    distance = np.min(dist, axis=1)
    if np.max(distance) > 1.8*half_width: return 0.
    widths = max(half_width, 1e-3)
    weights = np.exp(-.5*(dist/widths)**2)
    weights /= np.sum(weights, axis=1, keepdims=True)
    tangent = weights@(segments/lengths[:, None])
    increments = states[1:count+1, :2]-states[:count, :2]
    progress = np.sum(increments*tangent)
    return float(np.clip(progress, -target_speed*prediction_s, target_speed*prediction_s))
