"""Scene-independent evidence and conservative planning guard (NumPy only).

Raw network scores are never recalibrated or promoted. Evidence scores here are
heuristics, not probabilities. No scene identity, edited-object identity, or
simulator object ground truth is an input.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

try:
    from .box_geometry import BOX_CONVENTION, box_corners, points_to_box_xy
except ImportError:
    from box_geometry import BOX_CONVENTION, box_corners, points_to_box_xy


@dataclass(frozen=True)
class ReliabilityConfig:
    candidate_floor: float = 0.05
    background_mad_scale: float = 3.0
    confirmation_seconds: float = 0.2
    confirmation_observations: int = 3
    max_gap_seconds: float = 0.75
    association_distance_m: float = 2.0
    max_range_m: float = 60.0
    min_dimension_m: float = 0.2
    max_dimension_m: float = 30.0
    obstacle_margin_m: float = 0.5
    ego_half_width_m: float = 0.8
    ego_half_length_m: float = 1.5
    # Speed under unresolved, path-relevant uncertainty; physical units.
    caution_speed_mps: float = 2.0
    braking_mps2: float = 2.0
    plan_dt_seconds: float = 0.5

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(float(v)) and v > 0 for v in values.values()):
            raise ValueError("Reliability parameters must be finite and positive")
        if self.candidate_floor >= 1 or self.min_dimension_m >= self.max_dimension_m:
            raise ValueError("Invalid score/dimension bounds")


def _numpy(value):
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pose_from_row_rotation(rotation, translation):
    """UniAD l2g_r_mat uses row vectors; this module uses column vectors."""
    pose = np.eye(4)
    pose[:3, :3] = _numpy(rotation).reshape(3, 3).T
    pose[:3, 3] = _numpy(translation).reshape(3)
    return pose


def _validate_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
        raise ValueError("Invalid column-vector lidar-to-world pose")
    return pose


def _visible(box, metadata):
    """A center/corner in any supplied real camera suffices; no camera-count rule."""
    projections = metadata.get("lidar2img")
    shapes = metadata.get("img_shape")
    if projections is None or shapes is None or len(projections) != len(shapes):
        return None
    points = np.vstack([box[:3] + [0, 0, box[5]/2], box_corners(box)])
    points = np.column_stack([points, np.ones(len(points))])
    for projection, shape in zip(projections, shapes):
        p = np.asarray(projection, dtype=float)
        if p.shape != (4, 4) or not np.isfinite(p).all():
            raise ValueError("Invalid lidar2img projection")
        projected = points @ p.T
        z = projected[:, 2]
        xy = projected[:, :2] / np.maximum(z[:, None], 1e-6)
        if np.any((z > 1e-3) & (xy[:, 0] >= 0) & (xy[:, 0] < shape[1])
                  & (xy[:, 1] >= 0) & (xy[:, 1] < shape[0])):
            return True
    return False


def path_clearance(box, trajectory, config):
    """Conservative swept-path check including model velocity, never truth.

    UniAD boxes use a yaw-oriented width(x)/length(y) rectangle. The ego is
    bounded by a circle so its orientation/reference-point assumptions remain
    explicit. Box coordinates use the model's LiDAR frame.
    """
    trajectory = np.asarray(trajectory, dtype=float).reshape(-1, 2)
    path = np.vstack([np.zeros(2), trajectory])
    if not np.isfinite(path).all():
        raise ValueError("Nonfinite planning trajectory")
    dense, times = [], []
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        # Include every segment, even when model waypoints are far apart.
        # Rigid rotations must not change sample count through roundoff at an
        # exact multiple of the spacing (e.g. 1.0 vs 1.0000000000000002 m).
        n = max(2, int(np.ceil(np.linalg.norm(b-a)/.25-1e-10)) + 1)
        alpha = np.linspace(0, 1, n)
        dense.extend(a[None] + alpha[:, None] * (b-a))
        times.extend((i + alpha) * config.plan_dt_seconds)
    points = np.asarray(dense)
    velocity = box[7:9] if len(box) >= 9 else np.zeros(2)
    local = points_to_box_xy(points-np.asarray(times)[:, None]*velocity, box)
    outside = np.maximum(np.abs(local) - box[None, 3:5]/2, 0)
    radius = np.hypot(config.ego_half_width_m, config.ego_half_length_m)
    clearance = np.linalg.norm(outside, axis=1) - radius - config.obstacle_margin_m
    return float(clearance.min())


class SceneReliability:
    def __init__(self, config=None):
        self.config = ReliabilityConfig(**(config or {}))
        self.reset()

    def reset(self):
        self.scene_token = None
        self.previous_time = None
        self.tracks = []
        self.next_id = 0

    def update(self, boxes, scores, labels, timestamp, scene_token, pose=None,
               metadata=None, trajectory=None, detection_threshold=0.4):
        cfg = self.config
        boxes = _numpy(boxes).astype(float)
        scores = _numpy(scores).astype(float).reshape(-1)
        labels = _numpy(labels).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 7 or len(boxes) != len(scores) or len(labels) != len(scores):
            raise ValueError("Expected aligned boxes[N,>=7], scores[N], labels[N]")
        if (not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or not np.isfinite(labels).all() or not np.all(labels == labels.astype(int))):
            raise ValueError("Invalid network scores or class labels")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or not 0 < detection_threshold <= 1:
            raise ValueError("Invalid timestamp/detection threshold")
        reset_reason = None
        if scene_token != self.scene_token:
            reset_reason = "scene_changed"
        elif self.previous_time is not None and timestamp <= self.previous_time:
            reset_reason = "non_increasing_time"
        elif self.previous_time is not None and timestamp-self.previous_time > cfg.max_gap_seconds:
            reset_reason = "time_gap"
        if reset_reason:
            self.reset()
        self.scene_token, self.previous_time = scene_token, timestamp
        pose = _validate_pose(pose) if pose is not None else None
        if pose is None:
            self.tracks = []  # Never associate LiDAR coordinates across moving frames.
        metadata = metadata or {}
        # Edit-quality trust is independent of class-score uncertainty.
        # Shared module is also imported standalone by CPU replay scripts.
        try:
            from .edit_trust import assert_edit_evaluable, diagnostic_closed_loop_enabled
        except ImportError:
            from edit_trust import assert_edit_evaluable, diagnostic_closed_loop_enabled
        assert_edit_evaluable(metadata)
        if len(scores):
            median = float(np.median(scores))
            mad = float(np.median(np.abs(scores-median)))
            # Uses all query scores, before selecting objects/classes.
            threshold = max(cfg.candidate_floor, median + cfg.background_mad_scale * 1.4826 * mad)
        else:
            threshold = cfg.candidate_floor
        valid = (np.isfinite(boxes).all(axis=1) & (labels >= 0) & (labels < 10)
                 & (boxes[:, 3:6] >= cfg.min_dimension_m).all(axis=1)
                 & (boxes[:, 3:6] <= cfg.max_dimension_m).all(axis=1)
                 & (np.linalg.norm(boxes[:, :2], axis=1) <= cfg.max_range_m))
        if boxes.shape[1] >= 9:
            valid &= np.linalg.norm(boxes[:, 7:9], axis=1) <= 80.
        selected = np.flatnonzero(valid & ((scores >= threshold) | (scores >= detection_threshold)))
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        previous = [t for t in self.tracks if timestamp-t["time"] <= cfg.max_gap_seconds]
        used, new_tracks, obstacles, kept = set(), [], [], []
        for index in selected:
            box = boxes[index]
            visible = _visible(box, metadata)
            if visible is False:
                continue
            # Query duplicates cannot manufacture temporal confirmation.
            if any(np.linalg.norm(box[:2]-boxes[j, :2]) < min(1., min(box[3:5])/2) for j in kept):
                continue
            kept.append(index)
            world = pose[:3, :3] @ box[:3] + pose[:3, 3] if pose is not None else box[:3]
            velocity = np.zeros(3)
            if pose is not None and len(box) >= 9:
                velocity = pose[:3, :3] @ np.r_[box[7:9], 0.]
            possible = []
            for j, track in enumerate(previous):
                if j in used or track["label"] != int(labels[index]):
                    continue
                predicted = track["world"] + track["velocity"]*(timestamp-track["time"])
                distance = np.linalg.norm(world-predicted)
                ratio = box[3:6]/track["size"]
                if distance <= cfg.association_distance_m and np.all((ratio >= .5) & (ratio <= 2)):
                    possible.append((distance, j))
            if possible:
                _, match = min(possible)
                used.add(match)
                track = previous[match]
                identity, first, count = track["id"], track["first"], track["count"]+1
            else:
                identity, first, count = self.next_id, timestamp, 1
                self.next_id += 1
            new_tracks.append(dict(id=identity, first=first, count=count, time=timestamp,
                                   world=world, velocity=velocity, size=box[3:6].copy(), label=int(labels[index])))
            high = scores[index] >= detection_threshold
            persistent = (pose is not None and count >= cfg.confirmation_observations
                          and timestamp-first >= cfg.confirmation_seconds-1e-6)
            state = "network_retained" if high else ("persistent_uncertain" if persistent else "provisional_uncertain")
            obstacle = dict(candidate_index=int(index), evidence_track_id=identity,
                            box=box.tolist(), raw_score=float(scores[index]), label=int(labels[index]),
                            state=state, observations=count, duration_seconds=timestamp-first,
                            visible=visible, score_is_calibrated_probability=False)
            if trajectory is not None:
                obstacle["path_clearance_m"] = path_clearance(box, trajectory, cfg)
            obstacles.append(obstacle)
        self.tracks = new_tracks
        conflicts = [o for o in obstacles if o["state"] != "network_retained"
                     and o.get("path_clearance_m", float("inf")) <= 0]
        return dict(schema_version=1, policy="scene_independent_uncertainty_guard_v2_legacy_geometry",
                    box_convention=BOX_CONVENTION,
                    candidate_count=len(scores), invalid_geometry_count=int((~valid).sum()),
                    adaptive_evidence_threshold=threshold, detection_threshold=float(detection_threshold),
                    raw_scores_modified=False, confidence_calibrated=False,
                    edit_trust_verified=bool((metadata.get("edit_trust") or {}).get("edited")) and not diagnostic_closed_loop_enabled(),
                    diagnostic_closed_loop=diagnostic_closed_loop_enabled(),
                    config=asdict(cfg),
                    ground_truth_used=False, temporal_pose_available=pose is not None,
                    camera_visibility_available=metadata.get("lidar2img") is not None,
                    reset_reason=reset_reason, obstacles=obstacles,
                    uncertain_path_conflicts=len(conflicts),
                    action="caution" if conflicts else "unchanged",
                    speed_limit_mps=cfg.caution_speed_mps if conflicts else None,
                    braking_mps2=cfg.braking_mps2,
                    coverage="model proposals only; no guarantee for undetected objects")


def augment_occupancy(occupancy, report):
    """Add uncertain proposal footprints to an MPC copy, not neural occupancy.

    Five temporal masks are treated conservatively as a union by the existing
    MPC. Extrapolate the model velocity across its 3s horizon on that same basis.
    """
    result = np.asarray(occupancy).copy()
    if result.shape not in ((1, 5, 1, 200, 200), (1, 5, 200, 200)):
        raise ValueError("Unsupported occupancy shape")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite occupancy")
    margin = float(report.get("config", {}).get("obstacle_margin_m", ReliabilityConfig().obstacle_margin_m))
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid configured obstacle margin")
    yy, xx = np.mgrid[:200, :200]
    grid = np.stack([(xx-100)*.5+.25, (yy-100)*.5+.25], axis=-1)
    union = np.zeros((200, 200), dtype=bool)
    for obstacle in report.get("obstacles", []):
        if obstacle["state"] == "network_retained":
            continue
        box = np.asarray(obstacle["box"], dtype=float)
        velocity = box[7:9] if len(box) >= 9 else np.zeros(2)
        # Half-cell diagonal dilation closes raster gaps.
        for t in np.linspace(0, 3, max(7, int(np.ceil(np.linalg.norm(velocity)*3/.25))+1)):
            local = points_to_box_xy(grid-velocity*t, box)
            union |= (np.abs(local) <= box[3:5]/2 + margin + np.sqrt(2)*.25).all(axis=-1)
    masks = result.reshape(1, 5, 200, 200)
    masks[0, :, union] = 1
    return result


def limit_acceleration(acceleration, speed, dt, report):
    """Native FIFO controller counterpart of the MPC's reachable speed cap."""
    cap = report.get("speed_limit_mps") if report else None
    if cap is None:
        return float(acceleration)
    if dt <= 0 or not np.isfinite([acceleration, speed, dt, cap]).all():
        raise ValueError("Invalid speed-guard input")
    required = max(-float(report["braking_mps2"]), (float(cap)-speed)/dt)
    return float(min(acceleration, required))
