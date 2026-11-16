
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import torch
import hugsim_env
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as SCR
from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import bg_collision_det, ego_verts_canonic

def predictive_background_collision(plan_traj, info, sim, return_details=False):
    """Check future LTF poses against the runtime Gaussian point cloud."""
    if plan_traj is None or len(plan_traj) == 0:
        return (False, None) if return_details else False

    # LTF lidar frame: x-right, y-forward.
    imu_plan = np.asarray(plan_traj[:, [1, 0]], dtype=np.float32).copy()
    imu_plan[:, 1] *= -1

    global_traj = traj_transform_to_global(imu_plan, info["ego_box"])
    if len(global_traj) == 0:
        return (False, None) if return_details else False

    points = sim.points
    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    else:
        points = points.float()

    # Runtime HUGSIM coordinates [x_h, y_h, z_h] to scorer coordinates.
    points = torch.stack(
        [points[:, 2], -points[:, 0], -points[:, 1]],
        dim=1
    )

    points_full = points
    points = points_full[::8]

    _, _, z, width, length, height, _ = info["ego_box"]
    scale = np.array([length, width, height], dtype=np.float32)
    diagnostic = True
    max_full_count = 0
    max_sampled_count = 0
    max_pose_idx = -1
    sampled_collision = False
    full_collision = False
    first_full_hit_idx = None

    def count_box_points(cloud, box):
        O, A, B, C = box[0], box[1], box[2], box[5]
        OA, OB, OC = A - O, B - O, C - O
        POA = (cloud @ OA[..., None])[:, 0]
        POB = (cloud @ OB[..., None])[:, 0]
        POC = (cloud @ OC[..., None])[:, 0]
        mask = (
            (torch.dot(O, OA) < POA) & (POA < torch.dot(A, OA))
            & (torch.dot(O, OB) < POB) & (POB < torch.dot(B, OB))
            & (torch.dot(O, OC) < POC) & (POC < torch.dot(C, OC))
        )
        return int(mask.sum().item())

    for pose_idx, (x, y, yaw) in enumerate(global_traj):
        verts = ego_verts_canonic * scale
        rot = SCR.from_euler("z", float(yaw)).as_matrix()
        verts = (rot @ verts.T).T + np.array([x, y, z])
        verts = torch.as_tensor(
            verts, dtype=torch.float32, device=points.device
        )

        if diagnostic:
            full_count = count_box_points(points_full, verts)
            sampled_count = count_box_points(points, verts)
            if sampled_count > max_sampled_count:
                max_full_count = full_count
                max_sampled_count = sampled_count
                max_pose_idx = pose_idx

        sampled_hit = bg_collision_det(points, verts)
        full_hit = bg_collision_det(points_full, verts)
        if full_hit and first_full_hit_idx is None:
            first_full_hit_idx = pose_idx
        sampled_collision = sampled_hit or sampled_collision
        full_collision = full_hit or full_collision

    if diagnostic:
        print(
            "predictive_bg_diag",
            "ego_x=", round(float(info["ego_box"][0]), 3),
            "max_full_points=", max_full_count,
            "max_sampled_points=", max_sampled_count,
            "pose_idx=", max_pose_idx,
            "threshold=100",
            "sampled_hit=", sampled_collision,
            "full_hit=", full_collision,
            "first_full_hit_idx=", first_full_hit_idx,
            flush=True,
        )

    if return_details:
        return full_collision, first_full_hit_idx
    return full_collision

import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
from moviepy import ImageSequenceClip

def to_video(observations, output_path):
    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def create_gym_env(cfg, output):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}
    supervision_dir = os.path.join(output, 'supervision')
    os.makedirs(supervision_dir, exist_ok=True)

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        observations_save.append(obs['rgb'])
        infos_save.append(info)

        def to_numpy(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        route_xyz = env.unwrapped.ground_model[0][:, :3, 3]
        ego_xyz = np.asarray(info['ego_pos'], dtype=np.float32)
        route_xz = route_xyz[:, [0, 2]]
        ego_xz = ego_xyz[[0, 2]]
        yaw = float(info['ego_rot'][1])

        # Select the nearest route point that is not behind the vehicle.
        route_delta_global = route_xz - ego_xz
        forward_proj = (
            route_delta_global[:, 0] * np.sin(yaw)
            + route_delta_global[:, 1] * np.cos(yaw)
        )
        nearest = int(np.argmin(np.linalg.norm(route_delta_global, axis=1)))
        future_indices = np.flatnonzero(
            (np.arange(len(route_xz)) >= nearest) & (forward_proj >= 0.0)
        )
        start = int(future_indices[0]) if len(future_indices) else nearest

        future_route = route_xz[start:start + 128]
        if len(future_route) == 0:
            future_route = route_xz[-1:]
        if len(future_route) < 128:
            future_route = np.concatenate(
                [future_route, np.repeat(future_route[-1:], 128 - len(future_route), axis=0)],
                axis=0,
            )

        route_delta = future_route - ego_xz
        expert_route = np.column_stack((
            route_delta[:, 0] * np.sin(yaw) + route_delta[:, 1] * np.cos(yaw),
            -route_delta[:, 0] * np.cos(yaw) + route_delta[:, 1] * np.sin(yaw),
        )).astype(np.float32)

        np.savez_compressed(
            os.path.join(supervision_dir, f"frame_{cnt:06d}.npz"),
            rgb=np.asarray({k: to_numpy(v) for k, v in obs["rgb"].items()}, dtype=object),
            depth=np.asarray({k: to_numpy(v) for k, v in obs["depth"].items()}, dtype=object),
            ego_pos=np.asarray(info["ego_pos"], dtype=np.float32),
            ego_rot=np.asarray(info["ego_rot"], dtype=np.float32),
            ego_velo=np.float32(info["ego_velo"]),
            command=np.int32(info["command"]),
            expert_route=expert_route,
        )

        print('ego pose', info['ego_pos'])

        with open(obs_pipe, "wb") as pipe:
            pipe.write(pickle.dumps((obs, info)))
        with open(plan_pipe, "rb") as pipe:
            plan_traj = pickle.loads(pipe.read())

        reliability = {}
        if isinstance(plan_traj, dict):
            reliability = plan_traj.get('scene_reliability', {})
            plan_traj = plan_traj['trajectory']
            with open(os.path.join(supervision_dir, f"reliability_{cnt:06d}.json"), 'w') as f:
                json.dump(reliability, f, allow_nan=False)

        frame_info = info

        if plan_traj is not None:
            sample_path = os.path.join(supervision_dir, f"frame_{cnt:06d}.npz")
            with np.load(sample_path, allow_pickle=True) as saved:
                sample = {k: saved[k] for k in saved.files}
            plan_array = np.asarray(plan_traj, dtype=np.float32)
            sample["ltf_traj"] = np.column_stack(
                (plan_array[:, 1], -plan_array[:, 0])
            ).astype(np.float32)
            sample["collision"] = np.bool_(frame_info.get("collision", False))
            sample["rc"] = np.float32(frame_info.get("rc", 0.0))
            np.savez_compressed(sample_path, **sample)

            acc, steer_rate = traj2control(plan_traj, info)
            # Shared physical rule, independent of scene identity. Keep the
            # trajectory unchanged and constrain acceleration at the controller.
            if reliability.get('speed_limit_mps') is not None:
                pass  # Path is provided by the portable boundary above
                reliability_dir = Path(__file__).resolve().parent.parent / 'UniAD_SIM/projects/mmdet3d_plugin/uniad'
                if str(reliability_dir) not in sys.path:
                    sys.path.insert(0, str(reliability_dir))
                from scene_reliability import limit_acceleration
                acc = limit_acceleration(acc, float(info['ego_velo']), env.unwrapped.dt, reliability)

            predictive_collision, collision_pose_idx = predictive_background_collision(
                plan_traj, info, env.unwrapped, return_details=True
            )
            expert_path_global = np.concatenate(
                [ego_xz[None, :], route_xz[start:]], axis=0
            ).astype(np.float32)
            route_segment = np.linalg.norm(
                np.diff(expert_path_global, axis=0), axis=1
            )
            route_arc = np.concatenate(
                [np.array([0.0], dtype=np.float32), np.cumsum(route_segment)]
            )
            keep = np.concatenate(
                [np.array([True]), np.diff(route_arc) > 1e-4]
            )
            route_arc = route_arc[keep]
            expert_path_global = expert_path_global[keep]

            expert_speed = max(float(info["ego_velo"]), 1.0)
            expert_distances = (
                expert_speed * np.arange(1, 9, dtype=np.float32) * 0.5
            )
            expert_coverage_m = float(route_arc[-1])
            expert_required_m = float(expert_distances[-1])
            expert_horizon_ok = (
                len(route_arc) >= 2
                and expert_coverage_m >= expert_required_m
            )

            if len(route_arc) >= 2:
                expert_distances = np.clip(
                    expert_distances, 0.0, expert_coverage_m
                )
                expert_samples_global = np.column_stack([
                    np.interp(
                        expert_distances, route_arc, expert_path_global[:, axis]
                    )
                    for axis in range(2)
                ])
            else:
                expert_samples_global = np.repeat(
                    ego_xz[None, :], 8, axis=0
                )

            expert_delta = expert_samples_global - ego_xz
            expert_local = np.column_stack((
                expert_delta[:, 0] * np.sin(yaw)
                + expert_delta[:, 1] * np.cos(yaw),
                -expert_delta[:, 0] * np.cos(yaw)
                + expert_delta[:, 1] * np.sin(yaw),
            )).astype(np.float32)
            sample["expert_route_horizon"] = expert_local.astype(np.float32)
            sample["expert_route_horizon_distances"] = expert_distances.astype(np.float32)
            np.savez_compressed(sample_path, **sample)

            expert_plan = np.column_stack(
                [-expert_local[:, 1], expert_local[:, 0]]
            ).astype(np.float32)

            expert_hit = (
                predictive_background_collision(
                    expert_plan, info, env.unwrapped
                )
                if expert_horizon_ok
                else None
            )
            ltf_hit = predictive_background_collision(
                plan_traj, info, env.unwrapped
            )
            print(
                "trajectory_safety",
                "cnt=", cnt,
                "expert_hit=", expert_hit,
                "expert_horizon_ok=", expert_horizon_ok,
                "expert_coverage_m=", round(expert_coverage_m, 2),
                "expert_required_m=", round(expert_required_m, 2),
                "ltf_hit=", ltf_hit,
                flush=True,
            )
            action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            sample_path = os.path.join(supervision_dir, f"frame_{cnt:06d}.npz")
            with np.load(sample_path, allow_pickle=True) as saved:
                sample = {k: saved[k] for k in saved.files}
            sample["collision_next"] = np.bool_(info.get("collision", False))
            sample["rc_next"] = np.float32(info.get("rc", 0.0))
            np.savez_compressed(sample_path, **sample)
            cnt += 1
            print(
                "step_done",
                "cnt=", cnt,
                "terminated=", terminated,
                "truncated=", truncated,
                "rc=", info.get("rc"),
                "collision=", info.get("collision"),
                "reward=", reward,
            )
            done = terminated or truncated or cnt > 400

        else:  # AD Side Crushed
            done = True

        imu_plan_traj = plan_traj[:, [1, 0]]
        imu_plan_traj[:, 1] *= -1
        global_traj = traj_transform_to_global(imu_plan_traj, frame_info['ego_box'])

        save_data['frames'].append({
            'time_stamp': info['timestamp'],
            'is_key_frame': True,
            'ego_box': info['ego_box'],
            'obj_boxes': info['obj_boxes'],
            'obj_names': ['car' for _ in info['obj_boxes']],
            'planned_traj': {
                'traj': global_traj,
                'timestep': 0.5
            },
            'collision': info['collision'],
            'rc': info['rc']
        })

    with open(obs_pipe, "wb") as pipe:
        pipe.write(pickle.dumps('Done'))

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    to_video(observations_save, os.path.join(output, 'video.mp4'))
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    else:
        raise NotImplementedError
    
    process = launch(ad_path, args.ad_cuda, output)
    try:
        create_gym_env(cfg, output)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # # For debug
    # create_gym_env(cfg, output)
