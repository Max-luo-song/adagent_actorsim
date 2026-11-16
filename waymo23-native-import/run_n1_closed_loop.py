"""Shared source-metadata closed loop; edited cases require a valid N1 domain.

Never follows recorded future ego poses. The source route supplies navigation
only; executed motion is integrated from the selected MPC action every 0.1s.
"""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE/"waymo23-native-import"),
               str(BASE/"waymo23-native-import/waymo172_closed_loop_v1")]
from planning_guard import ObstacleMemory, RecoveryController
from n1_cross_scene_audit import CrossSceneProtocol, source_geometry
from shared_rules import validate_case_rules
from runtime_configuration import load_vehicle_config, preflight_source
from build_source_sequence import training_lidar_mount, recorded_kinematics
from edit_trust_adapter import asset_scene_binding, load_evaluation_domain, assert_edit_evaluable, digest
from closed172_common import (B, native_delta, virtual_global_pose, optimize,
                              atomic_json, atomic_pickle, wait_json, reference_check, emergency_stop_action)


def implementation_hash():
    files = [Path(__file__), BASE/"waymo23-native-import/planning_guard.py", BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad/scene_reliability.py",
             BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad/edit_trust.py",
             BASE/"waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py",
             BASE/"waymo23-native-import/waymo172_closed_loop_v1/closed172_common.py",
             BASE/"waymo23-native-import/live_env.py", BASE/"waymo23-native-import/waymo552_three.py",
             BASE/"waymo23-native-import/shared_rules.py", BASE/"waymo23-native-import/n1_cross_scene_audit.py",
             BASE/"waymo23-native-import/edit_trust_adapter.py", BASE/"waymo23-native-import/build_source_sequence.py",
             BASE/"code/HUGSIM/configs/sim/nuscenes_camera.yaml",
             BASE/"waymo23-native-import/runtime_configuration.py",
             BASE/"waymo23-native-import/configs/vehicle.json",
             BASE/"waymo23-native-import/run_reliable_sequence.py",
             BASE/"waymo23-native-import/uniad_single.py", BASE/"waymo23-native-import/infer_three_pad32.py"]
    files += [BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'/name
              for name in ('box_geometry.py', 'input_geometry.py')]
    files.append(BASE/'code/UniAD_SIM/tools/closeloop/dataparser.py')
    files.append(BASE/'waymo23-native-import/frozen_bev_experiment.py')
    files.append(BASE/'waymo23-native-import/asset_geometry.py')
    files.extend(BASE/'waymo23-native-import'/n for n in ('model_profiles.py','diagnostic_closed_loop.py','asset_radiance.py'))
    files += [BASE/'code/mmdetection3d-uniad-legacy/mmdet3d/core/bbox/structures'/name
              for name in ('lidar_box3d.py', 'utils.py')]
    return digest({str(p.relative_to(BASE)): p.read_text() for p in files})


def run(case_path, output, *, planning_bev="history", model_profile="legacy", diagnostic=False):
    from edit_trust import diagnostic_closed_loop_enabled
    if diagnostic and not diagnostic_closed_loop_enabled():
        raise ValueError("Diagnostic runner requires its explicit scoped context")
    case_path, output = Path(case_path).resolve(), Path(output).resolve()
    case = json.loads(case_path.read_text())
    protocol = replace(CrossSceneProtocol(), planning_bev=planning_bev)
    code_at_start = implementation_hash()
    rules_binding = validate_case_rules(case, planning_bev=planning_bev, model_profile=model_profile)
    source = json.loads(Path(case["source_report"]).read_text())
    source_binding = preflight_source(source)
    vehicle = load_vehicle_config()
    if not np.isclose(vehicle.dt, protocol.simulation_dt):
        raise ValueError("Vehicle and protocol simulation timesteps differ")
    geometry = source_geometry(source, protocol)
    asset = None
    domain = None
    if case.get("asset_report"):
        asset = json.loads(Path(case["asset_report"]).read_text())
        if not diagnostic and not case.get("edit_trust_state"):
            raise ValueError("Edited closed loop requires independently revalidated N1 evidence")
        if not diagnostic:
            domain = load_evaluation_domain(case["edit_trust_state"], asset_scene_binding(source["checkpoint"], asset))
    elif case.get("edit_trust_state") or case.get("scene_edits"):
        raise ValueError("Inconsistent edit declaration")
    rules_binding = validate_case_rules(case, domain, planning_bev=planning_bev, model_profile=model_profile)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"shared_rules.json", rules_binding)
    atomic_json(output/"scene_adapter.json", source_binding)
    os.environ["BASE"] = str(BASE)
    sys.path[:0] = [str(BASE/"code/HUGSIM"), str(BASE/"code/HUGSIM/sim")]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    import gaussian_renderer as renderer
    import waymo552_three as adapter
    from export_snapshot import WAYMO_TO_HUGSIM as T
    from scipy.spatial.transform import Rotation
    from closed172_run import draw_frames
    from PIL import Image
    mount = training_lidar_mount(BASE/"code/HUGSIM/configs/sim/nuscenes_camera.yaml")
    frame, fps = geometry["frame"], geometry["fps"]
    initial_pose = T@geometry["relative"][frame]@np.linalg.inv(T)
    began = time.monotonic()
    env = worker = worker_log = None
    original_render = renderer.render
    trace, frames, fronts = [], [], []
    status, error = "initializing", None
    initial_speed = None
    try:
        env = adapter.make_env(scale=1., source_report=case["source_report"], vehicle=vehicle)
        env.fps = fps
        if not np.isclose(env.motion.dt, protocol.simulation_dt) or not np.isclose(env.motion.wheelbase, vehicle.wheelbase_m):
            raise ValueError("Simulator and shared MPC mechanical assumptions differ")
        speed, steer, _ = recorded_kinematics(geometry["relative"], frame, fps, env.motion.wheelbase)
        initial_speed = speed
        env.motion.speed, env.motion.steer, env.motion.timestamp = speed, steer, frame/fps
        env.ego_pose = lambda: initial_pose@native_delta(env.motion.x, env.motion.z, env.motion.yaw)
        if asset is not None:
            if domain is not None:
                env.edit_trust_domain = domain
            from scene.obj_model import ObjModel
            with torch.no_grad():
                params, _ = torch.load(Path(asset["asset"])/"gs.pth", map_location="cuda", weights_only=False)
                car = ObjModel(int(params[0]), feat_mutable=False)
                car.restore(params, None)
                car._xyz = car._xyz.clone()
                car._xyz[:, 0] -= float(asset["local_recenter_xz"][0])
                car._xyz[:, 2] -= float(asset["local_recenter_xz"][1])
                placement = next(p for p in asset["placements"] if p["case"] == "heading_A")
                asset_pose = car._xyz.new_tensor(placement["asset_to_native"])
                if asset.get('radiance_frame') == 'asset_local':
                    from asset_radiance import sh_rotation
                    matrix = sh_rotation(np.asarray(placement['asset_to_native'])[:3,:3],int(params[0]))
                    coeff = car.get_features.detach()
                    car._features_rest = torch.einsum('ij,njc->nic', coeff.new_tensor(matrix[1:,1:]), coeff[:,1:,:]).contiguous()
                del params
            def edited_render(*args, **kwargs):
                camera, pc = kwargs["viewpoint"], kwargs["pc"]
                degree, rest, dynamics = pc.active_sh_degree, pc._features_rest, camera.dynamics
                pc._features_rest = pc._features_dc.new_zeros((len(pc._xyz), 15, 3))
                pc.active_sh_degree = 3
                camera.dynamics = dict(dynamics, library_car=asset_pose)
                kwargs["dynamic_gaussians"] = dict(kwargs.get("dynamic_gaussians", {}), library_car=car)
                try:
                    return original_render(*args, **kwargs)
                finally:
                    camera.dynamics = dynamics
                    pc.active_sh_degree, pc._features_rest = degree, rest
            renderer.render = edited_render

        def packet():
            with torch.no_grad():
                obs, info = adapter.observation(env)
            pose = env.ego_pose()
            source_pose = np.linalg.inv(T)@pose@T
            nearest = int(np.argmin(np.linalg.norm(geometry["relative"][:, :2, 3]-source_pose[:2, 3], axis=1)))
            command = recorded_kinematics(geometry["relative"], nearest, fps, env.motion.wheelbase)[2]
            virtual = virtual_global_pose(pose, mount)
            info.update(command=command, ego_pos=pose[:3, 3].tolist(),
                        edited_scene=asset is not None,
                        scene_token=digest(dict(source_report=case["source_report"], checkpoint=source["checkpoint"])),
                        simulation_dt=vehicle.dt,
                        ego_rot=Rotation.from_matrix(pose[:3, :3]).as_euler("XYZ").tolist(),
                        explicit_l2g_r_mat=virtual[:3, :3], explicit_l2g_t=virtual[:3, 3],
                        model_lidar_frame=dict(virtual_to_ego=mount.tolist()))
            for camera in info["cam_params"].values():
                camera["l2c"] = np.asarray(camera["l2c"])@mount
            if diagnostic:
                info["diagnostic_closed_loop"] = True
                info["edit_trust"] = dict(edited=asset is not None, status="diagnostic_unverified")
            assert_edit_evaluable(info)
            return obs, info

        obs, info = packet()
        if asset is not None and reference_check(env.ego_pose(), asset)["overlap"]:
            raise RuntimeError("Initial ego reference intersects inserted object")
        first = output/"step_000"
        first.mkdir()
        atomic_pickle(first/"observation.pkl", (obs, info))
        worker_env = dict(os.environ, BASE=str(BASE), FIVE_JOB=str(first), PYTHONUNBUFFERED="1",
                          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
            worker_env.pop(key, None)
        worker_log = (output/"uniad_worker.log").open("w")
        worker = subprocess.Popen([
            str(BASE/"tools/pixi-uniad-legacy/.pixi/envs/default/bin/python"),
            str(BASE/"waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py"),
            "--job", str(output), "--planning-bev", protocol.planning_bev,
            "--model-profile", model_profile]+(["--diagnostic-closed-loop"] if diagnostic else []),
            env=worker_env, cwd=BASE/"code/HUGSIM", stdout=worker_log, stderr=subprocess.STDOUT)
        print("Loading resident UniAD; log:", output/"uniad_worker.log", flush=True)
        wait_json(output/"ready.json", worker, timeout=900.)
        warm = None
        controller = RecoveryController(); obstacle_memory = ObstacleMemory()
        for index in range(protocol.closed_loop_steps):
            folder = output/f"step_{index:03d}"
            folder.mkdir(exist_ok=True)
            path = folder/"observation.pkl"
            if index:
                atomic_pickle(path, (obs, info))
            atomic_json(output/f"request_{index:03d}.json", dict(index=index, packet=str(path)))
            response = wait_json(output/f"done_{index:03d}.json", worker, timeout=300.)
            if not np.isclose(response["timestamp"], info["timestamp"]):
                raise RuntimeError("Worker timestamp mismatch")
            with np.load(response["perception"], allow_pickle=False) as perception:
                neural = perception["trajectory"].copy()
                occupancy = perception["occupancy"].copy()
                candidates = [perception["candidate_"+key].copy() for key in ("boxes","scores","labels")]
            reliability = json.loads(Path(response["reliability"]).read_text())
            reliability = obstacle_memory.update(reliability, *candidates, info)
            atomic_json(folder/'guard_memory_report.json', reliability)
            recovery_cap=controller.speed_cap(reliability)
            if recovery_cap is not None:
                previous_cap=reliability.get('speed_limit_mps')
                reliability=dict(reliability,speed_limit_mps=min(recovery_cap,previous_cap) if previous_cap is not None else recovery_cap)
            result=optimize(neural,occupancy,info,warm=warm,maxiter=220,reliability=reliability,vehicle=vehicle)
            atomic_json(folder/"optimization.json",result)
            action,decision=controller.update(result,env.motion.speed,info['timestamp'],vehicle.dt,vehicle.emergency_braking_mps2)
            atomic_json(folder/'control_decision.json',decision)
            following=decision['state_after'] in ('normal','caution','recovering')
            warm=np.asarray(result['selected']['controls']) if result['selected'] is not None and following else None
            control_mode='feasible_mpc' if following else 'emergency_braking_unresolved_occupancy'
            before = env.ego_pose().copy()
            before_time = env.motion.timestamp
            a, b = draw_frames(obs, ("DIAGNOSTIC N1 SUSPENDED | " if diagnostic else "")+"step=%d speed=%.2f" % (index, env.motion.speed))
            frames.append(a); fronts.append(b)
            env.motion.advance(action)
            obs, info = packet()
            check = reference_check(env.ego_pose(), asset) if asset is not None else None
            row = dict(step=index, timestamp_before=before_time, timestamp_after=env.motion.timestamp,
                       action=action, speed_after=env.motion.speed,
                       control_mode=control_mode, control_decision=decision, optimized_path_certified=False,
                       native_pose_before=before.tolist(), native_pose_after=env.ego_pose().tolist(),
                       neural_plan=neural.tolist(), reliability_action=reliability["action"],
                       speed_limit_mps=reliability["speed_limit_mps"],
                       inserted_reference_overlap=check["overlap"] if check else None,
                       inserted_reference_gap_m=check["gap_m"] if check else None)
            trace.append(row)
            atomic_json(output/"trace.json", trace)
            if index:
                path.unlink()  # retain initial input and all small prediction/decision records
            print("Executed", index+1, "speed", round(env.motion.speed, 3), reliability["action"], flush=True)
            if check and check["overlap"]:
                status = "inserted_vehicle_reference_overlap"
                break
        else:
            status = ("stopped_under_unresolved_occupancy" if controller.state not in ("normal","caution","recovering") and env.motion.speed <= 1e-8
                      else "requested_steps_completed")
        atomic_pickle(output/"final_observation.pkl", (obs, info))
    except BaseException as exc:
        status, error = "error", repr(exc)
        traceback.print_exc()
    finally:
        if worker is not None:
            atomic_json(output/"stop.json", dict(stop=True))
            try:
                worker.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    worker.kill(); worker.wait()
        if worker_log is not None:
            worker_log.close()
        renderer.render = original_render
        if env is not None:
            env.close()
        if frames:
            frames[0].save(output/"three_views.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
            fronts[0].save(output/"front.gif", save_all=True, append_images=fronts[1:], duration=100, loop=0)
        summary = dict(diagnostic_closed_loop=diagnostic, n1_validation_suspended=diagnostic, formal_evaluation_admitted=False, model_input_profile=model_profile, status=status, error=error, scene=source.get("scene"), protocol_hash=case["protocol_hash"],
                       start_frame=frame, initial_speed_mps=initial_speed, steps_executed=len(trace),
                       simulated_seconds=len(trace)*protocol.simulation_dt, wall_seconds=time.monotonic()-began,
                       edited_case=asset is not None, n1_edit_certified=domain is not None,
                       recorded_future_ego_poses_used_for_motion=False, navigation_uses_recorded_route=True,
                       planning_bev=protocol.planning_bev, model_truth_object_input=False,
                       detection_accuracy_validated=False, formal_hugsim_scoring=False,
                       body_reference_calibrated=False, other_obstacle_actual_collisions_validated=False,
                       generalization_proven=False)
        summary.update(rules_binding)
        summary["implementation_hash"] = code_at_start
        summary["emergency_braking_steps"] = sum(r.get("control_mode") == "emergency_braking_unresolved_occupancy" for r in trace)
        summary["emergency_collision_free_certified"] = False
        summary["implementation_unchanged_during_run"] = implementation_hash() == code_at_start
        atomic_json(output/"closed_loop_report.json", summary)
        atomic_json(output/"trace.json", trace)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if status in ("requested_steps_completed", "stopped_under_unresolved_occupancy") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.case).read_text())
    if request.get("experiment_type") == "diagnostic_closed_loop_v1":
        from diagnostic_closed_loop import run_pair
        raise SystemExit(run_pair(args.case, args.output))
    if request.get("experiment_type") == "frozen_bev_ab_v1":
        from frozen_bev_experiment import run_pair
        raise SystemExit(run_pair(args.case, args.output))
    raise SystemExit(run(args.case, args.output))
