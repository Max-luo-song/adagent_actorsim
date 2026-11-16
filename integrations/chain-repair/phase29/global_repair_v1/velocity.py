"""External reliability patch v2 (phase28, diagnostic only): depth-support velocity gate.

Rule (generic; no scene IDs, no fixed steps, no class lists, no ground truth):
  For each guard obstacle whose reported speed exceeds SPEED_GATE_MPS and whose
  claimed footprint should contain at least MIN_EXPECTED_VOXELS rendered-depth
  voxels (from the current frame's own observed-depth evidence), if the observed
  support ratio is below MIN_SUPPORT_RATIO, the detection has no surface support
  for its claimed motion. The patch keeps the box as a STATIC constraint but
  zeroes the velocity used for the guard's temporal expansion.

Rationale (measured, this case): a ghost bus echo (no annotation support,
4-step flickering track) carried a phantom 3.3 m/s reported velocity; the
guard's temporal expansion (planning_guard.clearance: temporal*speed) then
blocked even a full stop (brake-seed clearance -2.13m; zeroing that velocity
restores +1.40m). Real buses in the same frame show support ratios 0.44-0.60.

This inherited diagnostic gate keeps the static box but changes its temporal
expansion. Missing support is not proof of zero physical velocity; this gate
has not established global dynamic-object safety.
"""
import numpy as np

SPEED_GATE_MPS = 0.5
MIN_EXPECTED_VOXELS = 8.0
MIN_SUPPORT_RATIO = 0.05


def apply(reliability, info):
    """reliability: the guard input dict (mutated copy returned)."""
    rel = reliability
    points = np.asarray(rel.get('observed_depth_obstacles_xy') or [], dtype=float)
    geom = rel.get('observed_depth_geometry') or {}
    voxel = float(geom.get('voxel_resolution_m', 0.5))
    patched = []
    for obstacle in rel.get('obstacles', []):
        box = np.asarray(obstacle['box'], dtype=float)
        speed = float(np.linalg.norm(box[7:9])) if len(box) >= 9 else 0.0
        if speed <= SPEED_GATE_MPS or len(points) == 0:
            continue
        L, W = float(box[3]), float(box[4])
        expected = (L * W) / (voxel * voxel)
        if expected < MIN_EXPECTED_VOXELS:
            continue
        yaw = float(box[6])
        cs, sn = np.cos(yaw), np.sin(yaw)
        dxy = points[:, :2] - box[:2]
        xr = cs * dxy[:, 0] + sn * dxy[:, 1]
        yr = -sn * dxy[:, 0] + cs * dxy[:, 1]
        inside = int(((np.abs(xr) < L / 2) & (np.abs(yr) < W / 2)).sum())
        ratio = inside / expected
        if ratio >= MIN_SUPPORT_RATIO:
            continue
        box[7] = 0.0
        box[8] = 0.0
        obstacle['box'] = box.tolist()
        # neural_box is immutable original evidence, not the planner override.
        obstacle['guard_motion_box'] = box.tolist()
        obstacle['velocity_evidence_gate_v2'] = dict(
            zeroed_speed_mps=speed, depth_support_ratio=ratio,
            expected_voxels=expected, observed_voxels=inside)
        patched.append(dict(candidate_index=obstacle.get('candidate_index'),
                            evidence_track_id=obstacle.get('evidence_track_id'),
                            zeroed_speed_mps=speed, depth_support_ratio=ratio))
    if patched:
        rel['velocity_evidence_gate_v2_applied'] = patched
    return rel
