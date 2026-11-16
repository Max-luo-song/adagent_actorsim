"""Rigid-motion invariant metrics for corresponding rendered Gaussian centers."""
import numpy as np


def shape_residual(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 4:
        raise ValueError('need >=4 corresponding 3D centers')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite centers')
    x, y = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(u @ vt)
    residual = np.linalg.norm(x @ (u @ fix @ vt) - y, axis=1)
    return dict(rms_m=float(np.sqrt(np.mean(residual ** 2))),
                p95_m=float(np.percentile(residual, 95)),
                centroid_displacement_m=float(np.linalg.norm(b.mean(0)-a.mean(0))))
