"""Diagnostic closed loop: resident source renderer + resident legacy UniAD + MPC.

No source checkpoint or existing project script is modified. All truth geometry
is used only for recording/collision termination, never by the optimizer.
"""
import argparse
import copy
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw

from closed172_common import (B,atomic_json,atomic_pickle,wait_json,native_delta,
    initial_native_pose,virtual_global_pose,optimize,reference_check)


def draw_frames(obs,label):
    order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
    canvas = Image.new("RGB",(1440,348),"white")
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(order):
        canvas.paste(Image.fromarray(obs["rgb"][name]).resize((480,320)),(i*480,28))
        draw.text((i*480+5,8),label+" | "+name,fill="black")
    front = Image.fromarray(obs["rgb"]["CAM_FRONT"]).resize((720,480))
    single = Image.new("RGB",(720,508),"white")
    single.paste(front,(0,28)); ImageDraw.Draw(single).text((8,8),label,fill="black")
    return canvas,single


def render_summaries(job,trace,asset,frames,fronts):
    if frames:
        frames[0].save(job/"three_views_closed_loop.gif",save_all=True,
                       append_images=frames[1:],duration=100,loop=0)
        fronts[0].save(job/"front_closed_loop.gif",save_all=True,
                       append_images=fronts[1:],duration=100,loop=0)
    if not trace:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,8))
    from closed172_common import asset_reference_vertices
    target = asset_reference_vertices(asset)[:,[0,2]]
    ax.fill(target[:,0],target[:,1],color="gray",alpha=.7,label="Inserted car reference")
    centers = [np.asarray(trace[0]["native_pose_before"])[:3,3][[0,2]]]
    centers += [np.asarray(row["native_pose_after"])[:3,3][[0,2]] for row in trace]
    centers = np.asarray(centers)
    ax.plot(centers[:,0],centers[:,1],"o-",color="darkorange",markersize=3,
            label="Executed ego-origin trajectory")
    from closed172_common import body_vertices
    for row in trace[::max(1,len(trace)//12)]:
        vertices = body_vertices(np.asarray(row["native_pose_after"]))[:,[0,2]]
        vertices = np.vstack([vertices,vertices[0]])
        ax.plot(vertices[:,0],vertices[:,1],color="darkorange",alpha=.5)
    ax.set_aspect("equal",adjustable="datalim")
    ax.set_xlabel("Native world X (m)"); ax.set_ylabel("Native world Z (m)")
    ax.grid(alpha=.3);ax.legend(fontsize=8)
    ax.set_title("Executed trajectory; road legality not validated")
    fig.tight_layout();fig.savefig(job/"executed_trajectory.png",dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=150)
    parser.add_argument("--allow-animation-hold-last",action="store_true")
    parser.add_argument("--maxiter",type=int,default=220)
    parser.add_argument("--parity-max-error",type=int,default=0)
    parser.add_argument("--edit-trust-state", help="N1 locally revalidated trust-state JSON for this exact insertion")
    parser.add_argument("--planning-bev", choices=("history", "current"), default="history")
    parser.add_argument("--model-startup-timeout", type=float, default=900.)
    args = parser.parse_args()
    if not 1 <= args.steps <= 300:
        raise ValueError("steps must be 1..300")
    base = Path(os.environ["BASE"]).resolve()
    source = base/"outputs/waymo172_car_sequence_20261016_154023"
    bundle = Path(__file__).resolve().parent
    sys.path.insert(0,str(base/"waymo23-native-import"))
    sys.path[:0] = [str(base/"code/HUGSIM"), str(base/"code/HUGSIM/sim")]
    os.chdir(base/"code/HUGSIM")
    import torch
    import gaussian_renderer as renderer
    from scene.obj_model import ObjModel
    from waymo172_three import make_env,observation,SOURCE_REPORT

    sequence = json.loads((source/"sequence_report.json").read_text())
    asset = json.loads((source/"asset_report.json").read_text())
    first = next(row for row in sequence["frames"] if int(row["frame"]) == 165)
    with Path(first["packet"]).open("rb") as f:
        original_obs,initial_info = pickle.load(f)
    c = np.asarray(initial_info["model_lidar_frame"]["virtual_to_ego"],dtype=float)
    initial_pose = initial_native_pose(initial_info)
    job = base/"outputs"/("waymo172_true_closed_long_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    job.mkdir(parents=True)
    print("OUTPUT:",job,flush=True)
    env = None
    process = None
    log_file = None
    original_render = renderer.render
    trace,frames,fronts = [],[],[]
    reason = "initializing"
    fatal = None
    policy_events = []
    began = time.monotonic()
    initial_timestamp = float(initial_info["timestamp"])

    try:
        from edit_trust_adapter import EditTrustError, asset_scene_binding, load_evaluation_domain
        if not args.edit_trust_state:
            raise EditTrustError("N1 requires --edit-trust-state; an unverified inserted vehicle cannot enter evaluation")
        scene_source = json.loads((base/SOURCE_REPORT).read_text())
        edit_domain = load_evaluation_domain(args.edit_trust_state, asset_scene_binding(scene_source["checkpoint"], asset))
        env = make_env(scale=1.0)
        env.edit_trust_domain = edit_domain
        if not np.isclose(env.motion.dt,.1) or not np.isclose(env.motion.wheelbase,2.7):
            raise RuntimeError("Unexpected simulator dt/wheelbase")
        if not (np.isclose(float(env.action_space["acc"].low),-2.)
                and np.isclose(float(env.action_space["acc"].high),2.)
                and np.isclose(float(env.action_space["steer_rate"].low),-np.deg2rad(15.))
                and np.isclose(float(env.action_space["steer_rate"].high),np.deg2rad(15.))):
            raise RuntimeError("Current action bounds differ from validated experiment")
        if (initial_timestamp+args.steps*.1 > (env.frames-1)/env.fps+1e-6
                and not args.allow_animation_hold_last):
            raise ValueError("Requested run exceeds source animation range")
        env.motion.reset()
        env.motion.speed = float(initial_info["ego_velo"])
        env.motion.steer = float(initial_info["ego_steer"])
        env.motion.timestamp = initial_timestamp

        def ego_pose():
            m = env.motion
            return initial_pose @ native_delta(m.x,m.z,m.yaw)
        env.ego_pose = ego_pose

        with torch.no_grad():
            params,iteration = torch.load(Path(asset["asset"])/"gs.pth",
                                          map_location="cuda",weights_only=False)
            car = ObjModel(int(params[0]),feat_mutable=False)
            car.restore(params,None)
            car._xyz = car._xyz.clone()
            car._xyz[:,0] -= float(asset["local_recenter_xz"][0])
            car._xyz[:,2] -= float(asset["local_recenter_xz"][1])
            if int(params[0]) != 3:
                raise RuntimeError("Native asset SH degree differs")
            placement = next(p for p in asset["placements"] if p["case"] == "heading_A")
            asset_pose = car._xyz.new_tensor(placement["asset_to_native"])
            del params

        def render_with_asset(*render_args,**kwargs):
            camera,pc = kwargs["viewpoint"],kwargs["pc"]
            previous_degree,previous_rest = pc.active_sh_degree,pc._features_rest
            dynamics = camera.dynamics
            pc._features_rest = pc._features_dc.new_zeros((len(pc._xyz),15,3))
            pc.active_sh_degree = 3
            camera.dynamics = dict(dynamics,library_car=asset_pose)
            kwargs["dynamic_gaussians"] = dict(kwargs.get("dynamic_gaussians",{}),library_car=car)
            try:
                return original_render(*render_args,**kwargs)
            finally:
                camera.dynamics = dynamics
                pc.active_sh_degree,pc._features_rest = previous_degree,previous_rest
        from edit_trust_adapter import mark_unverified_edit, assert_edit_evaluable
        mark_unverified_edit(env, dict(kind="insert", asset=asset["asset"], status="legacy_insertion_without_N1_evidence"))
        renderer.render = render_with_asset

        def get_packet():
            with torch.no_grad():
                obs,info = observation(env)
            pose = ego_pose()
            virtual = virtual_global_pose(pose,c)
            info.update(command=2,edited_scene=True,ego_pos=pose[:3,3].tolist(),
                        ego_rot=Rotation.from_matrix(pose[:3,:3]).as_euler("XYZ").tolist(),
                        explicit_l2g_r_mat=virtual[:3,:3].copy(),
                        explicit_l2g_t=virtual[:3,3].copy(),
                        model_lidar_frame=copy.deepcopy(initial_info["model_lidar_frame"]),
                        source_frame=min(env.frames-1,int(np.floor(env.motion.timestamp*env.fps+1e-6))),
                        source_animation_held_at_last_frame=bool(
                            env.motion.timestamp > (env.frames-1)/env.fps+1e-6))
            for camera in info["cam_params"].values():
                camera["l2c"] = np.asarray(camera["l2c"]) @ c
            return obs,info

        obs,info = get_packet()
        assert_edit_evaluable(info)
        parity = {}
        for name in original_obs["rgb"]:
            a,b = np.asarray(obs["rgb"][name]),np.asarray(original_obs["rgb"][name])
            if a.shape != b.shape:
                raise RuntimeError("Initial resolution differs: "+name)
            delta = np.abs(a.astype(np.int16)-b.astype(np.int16))
            parity[name] = dict(max_error=int(delta.max()),mean_error=float(delta.mean()))
        atomic_json(job/"initial_render_parity.json",parity)
        print("首帧重渲染误差：",parity,flush=True)
        for name,rgb in obs["rgb"].items():
            Image.fromarray(rgb).save(job/("initial_"+name+".png"))
        if any(row["max_error"] > args.parity_max_error for row in parity.values()):
            raise RuntimeError("Initial rendered RGB differs; stopped before worker/control")
        initial_check = reference_check(ego_pose(),asset)
        if initial_check["overlap"]:
            raise RuntimeError("Initial ego reference box overlaps inserted vehicle")
        step0 = job/"step_000";step0.mkdir()
        atomic_pickle(step0/"observation.pkl",(obs,info))
        label = "t=0.0s speed=%.3fm/s" % info["ego_velo"]
        a,b = draw_frames(obs,label);frames.append(a);fronts.append(b)

        command = [str(base/"tools/pixi/bin/pixi"),"run","--manifest-path",
                   str(base/"tools/pixi-uniad-legacy/pixi.toml"),"python",
                   str(bundle/"closed172_worker.py"),"--job",str(job),
                   "--planning-bev",args.planning_bev]
        log_file = (job/"uniad_worker.log").open("w")
        process = subprocess.Popen(command,cwd=base/"code/HUGSIM",env=os.environ.copy(),
                                   stdout=log_file,stderr=subprocess.STDOUT)
        print("UniAD加载一次，等待就绪；日志：",job/"uniad_worker.log",flush=True)
        wait_json(job/"ready.json",process,timeout=args.model_startup_timeout)
        warm = None

        for index in range(args.steps):
            folder = job/f"step_{index:03d}";folder.mkdir(exist_ok=True)
            packet = folder/"observation.pkl"
            if index:
                atomic_pickle(packet,(obs,info))
            atomic_json(job/f"request_{index:03d}.json",dict(index=index,packet=str(packet)))
            response = wait_json(job/f"done_{index:03d}.json",process,timeout=300.)
            if not np.isclose(response["timestamp"],info["timestamp"]):
                raise RuntimeError("Worker response timestamp mismatch")
            with np.load(response["perception"]) as result:
                neural_plan = result["trajectory"].copy()
                occupancy = result["occupancy"].copy()
            reliability = json.loads(Path(response["reliability"]).read_text())
            optimized = optimize(neural_plan,occupancy,info,warm=warm,maxiter=args.maxiter,
                                 reliability=reliability)
            atomic_json(folder/"optimization.json",optimized)
            selected = optimized["selected"]
            if selected is None:
                reason = "optimizer_no_feasible_candidate"
                policy_events.append(dict(step=index,event=reason,timestamp=info["timestamp"]))
                print("停止：本步未找到通过约束的候选；不执行失败解。",flush=True)
                break
            controls = np.asarray(selected["controls"])
            action = dict(acc=float(controls[0,0]),steer_rate=float(controls[0,1]))
            if abs(action["acc"]) > 2.+1e-8 or abs(action["steer_rate"]) > math.radians(15)+1e-8:
                raise RuntimeError("Action outside declared bounds")

            # Forecast truth is saved as evaluation only, never fed back into MPC.
            forecasts = []
            before_pose = ego_pose()
            for state in np.asarray(selected["states"]):
                pose = before_pose @ native_delta(state[0]+c[0,3],state[1]+c[1,3],state[2])
                forecasts.append(reference_check(pose,asset))
            atomic_json(folder/"forecast_truth_evaluation.json",dict(
                minimum_gap_m=min(p["gap_m"] for p in forecasts),
                any_overlap=any(p["overlap"] for p in forecasts),
                optimizer_did_not_use_these_results=True))

            previous_motion = np.array([env.motion.x,env.motion.z,env.motion.yaw,
                                        env.motion.speed,env.motion.steer])
            before_time = float(info["timestamp"])
            env.motion.advance(action)
            after_motion = np.array([env.motion.x,env.motion.z,env.motion.yaw,
                                     env.motion.speed,env.motion.steer])
            if env.motion.speed < -1e-5:
                raise RuntimeError("Negative speed after action")
            after_pose = ego_pose()
            checks = []
            for u in np.linspace(0,1,11):
                state = previous_motion*(1-u)+after_motion*u
                checks.append(reference_check(initial_pose @ native_delta(state[0],state[1],state[2]),asset))
            actual = reference_check(after_pose,asset)
            obs,info = get_packet()
            row = dict(step=index,time_before=before_time,time_after=float(info["timestamp"]),
                       elapsed_sim_s=float(info["timestamp"])-initial_timestamp,
                       action=action,speed_after=float(env.motion.speed),
                       steer_after_rad=float(env.motion.steer),
                       native_pose_before=before_pose.tolist(),native_pose_after=after_pose.tolist(),
                       source_frame=info["source_frame"],
                       source_animation_held_at_last_frame=info.get("source_animation_held_at_last_frame",False),
                       neural_plan_lidar_xy=neural_plan.tolist(),
                       optimized_lidar_xy=np.asarray(selected["lidar"])[5::5].tolist(),
                       optimizer_seed=selected["seed"],
                       predicted_occupancy_clearance_lower_bound_m=selected["clearance_lower_bound_m"],
                       inserted_reference_overlap=any(p["overlap"] for p in checks),
                       inserted_reference_minimum_gap_m=min(p["gap_m"] for p in checks),
                       inserted_reference_check_after=actual)
            trace.append(row)
            atomic_json(job/"trace.json",trace)
            label = "t=%.1fs speed=%.3fm/s gap=%.2fm" % (
                row["elapsed_sim_s"],row["speed_after"],row["inserted_reference_minimum_gap_m"])
            a,b = draw_frames(obs,label);frames.append(a);fronts.append(b)
            print("step",index,"t",round(row["elapsed_sim_s"],1),"action",action,
                  "speed",round(row["speed_after"],3),
                  "reference_gap",round(row["inserted_reference_minimum_gap_m"],3),flush=True)
            warm = controls
            if row["inserted_reference_overlap"]:
                reason = "inserted_vehicle_reference_overlap"
                print("停止：近似车身参考框与插入车重叠。",flush=True)
                break
        else:
            reason = "requested_control_steps_completed"
        atomic_pickle(job/"final_observation.pkl",(obs,info))
    except BaseException as exc:
        reason = "error"
        fatal = repr(exc)
        traceback.print_exc()
    finally:
        atomic_json(job/"stop.json",dict(stop=True))
        if process is not None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
        if log_file is not None:
            log_file.close()
        renderer.render = original_render
        if env is not None:
            env.close()
        atomic_json(job/"trace.json",trace)
        try:
            render_summaries(job,trace,asset,frames,fronts)
        except Exception as exc:
            policy_events.append(dict(event="summary_render_error",error=repr(exc)))
        summary = dict(status=reason,error=fatal,steps_executed=len(trace),
                       simulated_seconds=len(trace)*.1,requested_steps=args.steps,
                       wall_seconds=time.monotonic()-began,start_frame=165,
                       scene_checkpoint_loads=int(env is not None),
                       uniad_worker_ready=(job/"ready.json").exists(),
                       camera_count=3,virtual_rear_views=False,
                       asset_static_in_world=True,recorded_future_ego_poses_used=False,
                       perception_bev="history",planning_bev=args.planning_bev,
                       original_collision_postprocessor="replaced by external motion-constrained MPC",
                       mpc_first_action_executed_seconds=.1,model_weights_modified=False,
                       perception_update_hz=10,training_domain_and_cadence_validated=False,
                       source_motion_fps_assumption=10.,
                       animation_tail_policy="hold last frame" if args.allow_animation_hold_last else "stop at source end",
                       ground_truth_used_by_optimizer=False,
                       truth_use="approximate inserted-vehicle reference evaluation and stop only",
                       other_obstacle_actual_collisions_validated=False,
                       ego_origin_equals_body_center_assumed=True,
                       scene_ground_model="fixed initial tangent plane for ego motion",
                       physical_steering_angle_limit_validated=False,
                       drivable_area_validated=False,formal_hugsim_scoring_enabled=False,
                       avoidance_validated=False,events=policy_events)
        atomic_json(job/"closed_loop_report.json",summary)
        print(json.dumps(summary,indent=2),flush=True)
        print("OUTPUT:",job,flush=True)
        if (job/"three_views_closed_loop.gif").exists():
            print("三视角视频：",job/"three_views_closed_loop.gif",flush=True)
        if (job/"executed_trajectory.png").exists():
            print("实际轨迹：",job/"executed_trajectory.png",flush=True)
    if fatal is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
