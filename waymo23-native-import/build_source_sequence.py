"""Render unedited source sequences with one metadata-driven camera/pose rule.

Recorded poses are used only for this open-loop baseline. This is not an edited
scene, not N1 certification, and not a feedback closed loop.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]


def preview_opt_in(source):
    requested = source.get('perception_preview_requested', False)
    if requested is not True and requested is not False:
        raise ValueError('Perception preview opt-in must be a boolean')
    if requested and not source.get('diagnostic_asset'):
        raise ValueError('Preview opt-in requires a declared diagnostic asset')
    return requested


def training_lidar_mount(path):
    import yaml
    from scipy.spatial.transform import Rotation
    cfg = yaml.safe_load(Path(path).read_text())
    basis = np.eye(4)
    basis[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    def transform(values, prefix):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("XYZ", values[prefix+"_rot"], degrees=True).as_matrix()
        result[:3, 3] = values[prefix+"_trans"]
        return result
    translations = []
    for name in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"):
        e = cfg["cams"][name]["extrinsics"]
        translations.append((basis@np.linalg.inv(transform(e, "v2c"))@transform(e, "l2c"))[:3, 3])
    if not np.allclose(translations, translations[0], atol=1e-5):
        raise ValueError("Pretrained virtual LiDAR mount is inconsistent across cameras")
    mount = np.eye(4)
    mount[:3, 3] = translations[0]
    return mount


def recorded_kinematics(relative, frame, fps, wheelbase):
    """Shared sensor-metadata derivative rule; no source-specific steering."""
    lo, hi = max(0, frame-1), min(len(relative)-1, frame+1)
    if lo == hi:
        raise ValueError("At least two poses are needed")
    speed = np.linalg.norm(relative[hi, :2, 3]-relative[lo, :2, 3])*fps/(hi-lo)
    heading = np.arctan2(relative[[lo, hi], 1, 0], relative[[lo, hi], 0, 0])
    delta = math.atan2(math.sin(heading[1]-heading[0]), math.cos(heading[1]-heading[0]))
    steer = math.atan(-delta*fps/(hi-lo)*wheelbase/max(speed, .1))
    future = min(len(relative)-1, frame+int(round(2*fps)))
    direction = relative[future, :2, 3]-relative[frame, :2, 3]
    forward = relative[frame, :2, 0]
    angle = math.atan2(forward[0]*direction[1]-forward[1]*direction[0], forward@direction)
    command = 1 if angle > .25 else (0 if angle < -.25 else 2)
    return float(speed), float(steer), command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.start_frame < 0 or not 1 <= args.frames <= 200 or args.frame_stride < 1:
        parser.error("Invalid frame selection")
    source_path = Path(args.source_report).resolve()
    source = json.loads(source_path.read_text())
    diagnostic_inference_requested = preview_opt_in(source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["BASE"] = str(BASE)
    os.environ["MPLCONFIGDIR"] = str(BASE/"outputs/gpu_test_service/matplotlib_cache")
    print("Source metadata checked; importing CUDA renderer...", flush=True)
    sys.path[:0] = [str(BASE/"code/HUGSIM"), str(BASE/"code/HUGSIM/sim")]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    import waymo552_three as adapter
    from export_snapshot import WAYMO_TO_HUGSIM as T
    from scipy.spatial.transform import Rotation
    inputs = Path(source["inputs"])
    start = int(source.get("start_timestep", 0))
    files = [p for p in sorted((inputs/"ego_pose").glob("*.txt")) if int(p.stem) >= start]
    poses = np.stack([np.loadtxt(p) for p in files[:int(source["frames"])]])
    relative = np.linalg.inv(poses[0])[None]@poses
    selected = list(range(args.start_frame, args.start_frame+args.frames*args.frame_stride, args.frame_stride))
    if selected[-1] >= len(relative):
        raise ValueError("Requested baseline exceeds source pose range")
    mount = training_lidar_mount(BASE/"code/HUGSIM/configs/sim/nuscenes_camera.yaml")
    basis = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    fps = float(source.get("pose_fps_assumption", 10.))
    report = dict(scene=source.get("scene"), source_report=str(source_path), frames=[],
                  scene_edits=[], edit_trust=dict(edited=False), full_recorded_ego_poses_used=True,
                  closed_loop_tested=False, n1_edit_certified=False, source_fps_assumption=fps,
                  virtual_to_ego=mount.tolist(), observation_sampling_hz=fps/args.frame_stride,
                  navigation="shared route-heading rule from recorded source route")
    from model_profiles import bind_sequence_model_profile
    bind_sequence_model_profile(source, report)
    env = None
    restore_renderer = None
    try:
        env = adapter.make_env(scale=1., source_report=source_path)
        env.fps = fps
        env.capture_edit_evidence = True
        if source.get("diagnostic_asset"):
            from library_asset_preview import install
            restore_renderer = install(env, source["diagnostic_asset"], output)
            report.update(scene_edits=[source["diagnostic_asset"]],
                          edit_trust=dict(edited=True, status="quarantined"),
                          diagnostic_preview_only=True,
                          perception_preview_requested=diagnostic_inference_requested)
        for frame in selected:
            native = T@relative[frame]@np.linalg.inv(T)
            env.ego_pose = lambda pose=native: pose.copy()
            env.motion.timestamp = frame/fps
            speed, steer, command = recorded_kinematics(relative, frame, fps, env.motion.wheelbase)
            env.motion.speed, env.motion.steer = speed, steer
            with torch.no_grad():
                obs, info = adapter.observation(env)
            if restore_renderer is not None:
                from edit_trust_adapter import runtime_edit_metadata
                info.update(edited_scene=True, edit_trust=runtime_edit_metadata(env))
            virtual = basis@native@np.linalg.inv(basis)@mount
            info.update(command=command, ego_pos=native[:3, 3].tolist(),
                        ego_rot=Rotation.from_matrix(native[:3, :3]).as_euler("XYZ").tolist(),
                        explicit_l2g_r_mat=virtual[:3, :3], explicit_l2g_t=virtual[:3, 3],
                        source_frame=frame, model_lidar_frame=dict(virtual_to_ego=mount.tolist()))
            for camera in info["cam_params"].values():
                camera["l2c"] = np.asarray(camera["l2c"])@mount
            folder = output/f"frame_{frame:03d}"
            folder.mkdir()
            evidence = {}
            for name, view in env._edit_validation_views.items():
                for key in ("rgb", "depth", "alpha", "K", "camera_to_world"):
                    evidence[name+"__"+key] = view[key]
            np.savez_compressed(folder/"render_evidence.npz", **evidence)
            if restore_renderer is not None:
                from PIL import Image
                for name, rgb in obs["rgb"].items():
                    Image.fromarray(rgb).save(folder/(name+".png"))
            packet = folder/"observation.pkl"
            with packet.open("wb") as stream:
                pickle.dump((obs, info), stream, protocol=4)
            report["frames"].append(dict(frame=frame, packet=str(packet), timestamp=frame/fps,
                                         speed_mps=speed, command=command))
            (output/"sequence_report.json").write_text(json.dumps(report, indent=2))
            print("Rendered source frame", frame, "command", command, flush=True)
    finally:
        if restore_renderer is not None:
            restore_renderer()
        if env is not None:
            env.close()
    print("SEQUENCE:", output/"sequence_report.json", flush=True)


if __name__ == "__main__":
    main()
