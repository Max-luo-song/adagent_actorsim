"""Build an immutable shared test protocol and inspect independent sensor support.

This is an evidence audit, not an edited-scene certificate or a driving success
claim. All placement/selection rules are shared across scenes.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"))
from edit_trust import digest


@dataclass(frozen=True)
class CrossSceneProtocol:
    source_fps: float = 10.
    min_start_speed_mps: float = 2.
    max_start_speed_mps: float = 8.
    lookahead_m: float = 12.
    closed_loop_steps: int = 30
    simulation_dt: float = .1
    candidate_ground_radius_m: float = 5.
    sensor_time_offsets: tuple = (-2, -1, 0, 1, 2)
    ground_inlier_m: float = .10
    ground_max_slope_degrees: float = 20.
    minimum_ground_points: int = 30
    track_start_score: float = .4
    track_filter_score: float = .35
    planning_bev: str = "history"
    negative_cases: tuple = ("floating", "penetrating", "single_view_damage", "unobserved_erasure")


def source_geometry(source, protocol):
    inputs = Path(source["inputs"])
    start = int(source.get("start_timestep", 0))
    pose_files = sorted((inputs/"ego_pose").glob("*.txt"))
    pose_files = [p for p in pose_files if int(p.stem) >= start]
    count = int(source.get("frames", len(pose_files)))
    poses = np.stack([np.loadtxt(p) for p in pose_files[:count]])
    if poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid source ego poses")
    relative = np.linalg.inv(poses[0])[None]@poses
    xy = relative[:, :2, 3]
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    fps = float(source.get("pose_fps_assumption", protocol.source_fps))
    speed = segment*fps
    arc = np.r_[0., np.cumsum(segment)]
    needed_frames = int(np.ceil(protocol.closed_loop_steps*protocol.simulation_dt*fps))
    valid = np.flatnonzero((speed >= protocol.min_start_speed_mps)
                          & (speed <= protocol.max_start_speed_mps)
                          & ((arc[-1]-arc[:-1]) >= protocol.lookahead_m)
                          & (np.arange(len(speed))+needed_frames < len(poses)))
    if not len(valid):
        raise ValueError("No start satisfies the shared duration/speed/coverage rule")
    frame = int(valid[0])
    distance = arc[frame]+protocol.lookahead_m
    anchor = np.array([np.interp(distance, arc, relative[:, axis, 3]) for axis in range(3)])
    j = max(frame, int(np.searchsorted(arc, distance)-1))
    tangent = xy[min(j+1, len(xy)-1)]-xy[j]
    return dict(frame=frame, fps=fps, speed_mps=float(speed[frame]), anchor=anchor,
                heading=float(np.arctan2(tangent[1], tangent[0])), relative=relative,
                raw_start=start, remaining_m=float(arc[-1]-arc[frame]))


def fit_observed_ground(points, threshold, max_slope_degrees=20., seed=0):
    """RANSAC on already sensor-labelled ground; never Gaussian road nodes."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("Insufficient independent ground points")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(80):
        p = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        scale = np.linalg.norm(normal)
        if scale < 1e-8:
            continue
        normal /= scale
        if abs(normal[2]) < np.cos(np.deg2rad(max_slope_degrees)):
            continue
        # Avoid starting a BLAS thread team for every three-column RANSAC
        # product; local placement search evaluates many small plane fits.
        good = np.abs(np.sum((points-p[0])*normal, axis=1)) <= threshold
        if good.sum() > best.sum():
            best = good
    if best.sum() < 3:
        raise ValueError("No stable observed ground plane")
    center = points[best, :2].mean(axis=0)
    design = np.column_stack([points[best, :2]-center, np.ones(best.sum())])
    plane = np.linalg.lstsq(design, points[best, 2], rcond=None)[0]
    residual = points[best, 2]-design@plane
    return dict(origin_xy=center.tolist(), plane=plane.tolist(), inliers=int(best.sum()),
                total=len(points), rms_m=float(np.sqrt(np.mean(residual**2))),
                slope_degrees=float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))))


def audit_source(source_path, protocol, output):
    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text())
    geometry = source_geometry(source, protocol)
    data = Path(source.get("data_dir", source.get("original_data", source["inputs"])))
    samples, non_ground, provenance = [], [], []
    for offset in protocol.sensor_time_offsets:
        f = geometry["frame"]+offset
        if not 0 <= f < len(geometry["relative"]):
            continue
        path = data/"lidar"/f'{geometry["raw_start"]+f:03d}.bin'
        if not path.is_file():
            continue
        if path.stat().st_size % (14*4):
            raise ValueError("Lidar schema differs from archived Nx14 float32 loader")
        raw = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 14)
        xyz = np.asarray(raw[:, 3:6])
        pose = geometry["relative"][f]
        xyz = xyz@pose[:3, :3].T+pose[:3, 3]
        near = np.linalg.norm(xyz[:, :2]-geometry["anchor"][:2], axis=1) <= protocol.candidate_ground_radius_m
        valid = near & np.isfinite(xyz).all(axis=1)
        ground = valid & (raw[:, 10] == 1)
        samples.append(xyz[ground])
        non_ground.append(xyz[valid & (raw[:, 10] == 0)])
        provenance.append(dict(path=str(path), bytes=path.stat().st_size,
                               source_frame=f, nearby_ground_points=int(ground.sum())))
    ground = np.concatenate(samples) if samples else np.empty((0, 3))
    occupied = np.concatenate(non_ground) if non_ground else np.empty((0, 3))
    if len(ground) < protocol.minimum_ground_points:
        raise ValueError("Candidate lacks independent LiDAR ground support")
    fit = fit_observed_ground(ground, protocol.ground_inlier_m, protocol.ground_max_slope_degrees)
    anchor = geometry["anchor"].copy()
    anchor[2] = (anchor[:2]-fit["origin_xy"])@np.asarray(fit["plane"][:2])+fit["plane"][2]
    np.savez_compressed(output/"sensor_skin.npz", ground_points=ground, occupied_points=occupied,
                        anchor_source_xyz=anchor)
    return dict(source_report=str(source_path), scene=source.get("scene"),
                checkpoint=source["checkpoint"], inputs=source["inputs"], sensor_data=str(data),
                automatic_start_frame=geometry["frame"], estimated_start_speed_mps=geometry["speed_mps"],
                source_fps=geometry["fps"], source_fps_independently_validated=False,
                remaining_recorded_distance_m=geometry["remaining_m"],
                proposed_anchor_source_xyz=anchor.tolist(), heading_source_radians=geometry["heading"],
                ground_fit=fit, sensor_provenance=provenance,
                support_source_kind="sensor", baseline_trust_complete=False,
                edit_certified=False, closed_loop_run=False,
                blockers=["independent edited-asset appearance/depth references not supplied",
                          "sensor support is not yet a complete multi-view trusted-domain field"],
                gpu_baseline_request=dict(action="build_source_sequence", source_report=str(source_path),
                                          start_frame=geometry["frame"], frames=3, frame_stride=5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = CrossSceneProtocol()
    policy = asdict(protocol)
    report = dict(protocol=policy, protocol_hash=digest(policy), sources=[],
                  generalized_success_proven=False, actual_edited_vehicle_recall_measured=False)
    for index, path in enumerate(args.source_reports):
        folder = output/f"source_{index:02d}"
        folder.mkdir()
        try:
            item = audit_source(path, protocol, folder)
            item["status"] = "sensor_audit_complete_edit_evidence_incomplete"
        except Exception as exc:
            item = dict(source_report=str(Path(path).resolve()), status="audit_failed", error=repr(exc))
        item["protocol_hash"] = report["protocol_hash"]
        (folder/"audit_report.json").write_text(json.dumps(item, indent=2, allow_nan=False))
        report["sources"].append(item)
    (output/"audit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
