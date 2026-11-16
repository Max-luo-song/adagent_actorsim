"""JSON reference -> existing HUGSIM iLQR -> shared limits -> native Motion.

The feedback is measured simulation state, not UniAD perception. RGB/depth are
rendered at executed poses. Target waypoints never overwrite the Motion state.
"""

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

import json
import math
import time
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
from ego_trajectory import number,ffmpeg_path,trajectory_metrics,draw_trajectory_panel
from scene_api import validate_action


def execution_plan(compiled,vehicle,args):
    dt=float(vehicle.dt)
    if not np.isclose(dt,1/compiled['fps'],rtol=0,atol=1e-9):
        raise ValueError('execution requires scene frame interval equal to vehicle dt')
    poses=np.asarray([sample.get('motion_ego_to_world',sample['ego_to_world']) for sample in compiled['samples']])
    anchor=poses[0];local=np.linalg.inv(anchor)[None]@poses
    # Native Motion is planar. Do not silently discard 3D path changes.
    if np.max(np.abs(local[:,:3,2]-[0,0,1]))>1e-5 or np.max(np.abs(local[:,2,3]))>1e-4:
        raise ValueError('execution supports a planar trajectory in its initial ego frame; use preview for 3D paths')
    states=np.zeros((len(poses),5))
    states[:,0]=local[:,0,3];states[:,1]=-local[:,1,3]
    states[:,2]=np.unwrap(-np.arctan2(local[:,1,0],local[:,0,0]))
    speeds=np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1)/dt
    states[:-1,3]=speeds;states[-1,3]=speeds[-1]
    if np.any(speeds>vehicle.max_speed_mps+1e-6):raise ValueError('reference speed exceeds shared vehicle limit')
    # Fit steering feedforward from the reference heading rate, in right-positive Motion axes.
    yaw_rate=np.diff(states[:,2])/dt
    states[:-1,4]=np.arctan2(vehicle.wheelbase_m*yaw_rate,np.maximum(speeds,1e-8))
    states[-1,4]=states[-2,4]
    speed=number(args.get('initial_speed_mps',speeds[0]));steer=number(args.get('initial_steer_rad',0.))
    if not 0<=speed<=vehicle.max_speed_mps:raise ValueError('initial speed exceeds shared vehicle limits')
    if abs(steer)>math.radians(vehicle.max_steer_degrees):raise ValueError('initial steering exceeds shared vehicle limits')
    tolerance=number(args.get('max_tracking_error_m',1.))
    if tolerance<=0:raise ValueError('max_tracking_error_m must be positive')
    heading_tolerance=number(args.get('max_tracking_heading_error_degrees',10.))
    if not 0<heading_tolerance<=180:raise ValueError('heading tolerance must be in (0,180]')
    return dict(anchor=anchor,reference_states=states,initial_speed_mps=speed,initial_steer_rad=steer,
        max_tracking_error_m=tolerance,max_tracking_heading_error_degrees=heading_tolerance,dt=dt)


def make_solver(vehicle):
    from sim.ilqr.lqr_solver import ILQRSolver,ILQRSolverParameters,ILQRWarmStartParameters
    params=ILQRSolverParameters(discretization_time=vehicle.dt,
        state_cost_diagonal_entries=[5.,5.,10.,1.,.1],input_cost_diagonal_entries=[.2,.5],
        state_trust_region_entries=[1.]*5,input_trust_region_entries=[1.]*2,
        max_ilqr_iterations=40,convergence_threshold=1e-6,max_solve_time=.05,
        max_acceleration=max(vehicle.max_acc_mps2,-vehicle.min_acc_mps2),
        max_steering_angle=min(math.radians(vehicle.max_steer_degrees),math.pi/3),
        max_steering_angle_rate=math.radians(vehicle.max_steer_rate_degrees),
        min_velocity_linearization=.01,wheelbase=vehicle.wheelbase_m)
    warm=ILQRWarmStartParameters(k_velocity_error_feedback=.5,k_steering_angle_error_feedback=.05,
        lookahead_distance_lateral_error=15.,k_lateral_error=.1,
        jerk_penalty_warm_start_fit=1e-4,curvature_rate_penalty_warm_start_fit=1e-2)
    return ILQRSolver(solver_params=params,warm_start_params=warm)


def clip_action(acc,rate,motion,vehicle):
    if not np.isfinite([acc,rate]).all():raise ValueError('iLQR returned nonfinite controls')
    dt=vehicle.dt;maxsteer=math.radians(vehicle.max_steer_degrees);maxrate=math.radians(vehicle.max_steer_rate_degrees)
    acc=float(np.clip(acc,max(vehicle.min_acc_mps2,-motion.speed/dt),
                      min(vehicle.max_acc_mps2,(vehicle.max_speed_mps-motion.speed)/dt)))
    rate=float(np.clip(rate,max(-maxrate,(-maxsteer-motion.steer)/dt),
                       min(maxrate,(maxsteer-motion.steer)/dt)))
    return validate_action(dict(acc_mps2=acc,steer_rate_radps=rate),vehicle,motion.speed,motion.steer)


def motion_state(motion):
    # Controller: x forward / y right / heading right-positive.
    return np.asarray([motion.z,motion.x,motion.yaw,motion.speed,motion.steer],dtype=float)


def motion_pose(anchor,motion):
    c,s=math.cos(motion.yaw),math.sin(motion.yaw)
    delta=np.array([[c,s,0,motion.z],[-s,c,0,-motion.x],[0,0,1,0],[0,0,0,1]])
    return anchor@delta


class TrackingDriver:
    def __init__(self,plan,vehicle,solver=None):
        from live_env import Motion
        self.vehicle=vehicle;self.plan=plan
        self.motion=Motion(dt=vehicle.dt,wheelbase=vehicle.wheelbase_m,start_speed=plan['initial_speed_mps'])
        self.motion.steer=plan['initial_steer_rad']
        self.solver=make_solver(vehicle) if solver is None else solver

    def advance(self,index):
        before=motion_state(self.motion);references=self.plan['reference_states'][index:index+21].copy()
        supplied=len(references)
        # Existing iLQR warm-start fits derivatives and requires >=3 poses.
        # Solver-only extension maintains terminal speed/curvature; it never
        # advances the simulation or renders beyond the supplied trajectory.
        while len(references)<6:
            last=references[-1].copy();nxt=last.copy();dt=self.vehicle.dt
            nxt[0]+=last[3]*math.cos(last[2])*dt;nxt[1]+=last[3]*math.sin(last[2])*dt
            nxt[2]+=last[3]*math.tan(last[4])/self.vehicle.wheelbase_m*dt
            references=np.vstack([references,nxt])
        began=time.perf_counter();solutions=self.solver.solve(before.copy(),references.copy())
        if not solutions:raise RuntimeError('iLQR returned no solution; no action executed')
        raw=np.asarray(solutions[-1].input_trajectory[0],dtype=float)
        if raw.shape!=(2,):raise ValueError('iLQR returned invalid control shape')
        action=clip_action(raw[0],raw[1],self.motion,self.vehicle)
        native=dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps'])
        self.motion.advance(native)
        after=motion_state(self.motion)
        if not np.isfinite(after).all():raise RuntimeError('Motion produced nonfinite state')
        return dict(index=index,action=action,raw_control=raw.tolist(),state_before=before.tolist(),
            state_after=after.tolist(),solve_seconds=time.perf_counter()-began,
            reference_samples=len(references),supplied_reference_samples=supplied,solver_only_terminal_extension=len(references)-supplied,vehicle_limits_clipped=bool(np.max(np.abs(raw-np.asarray(list(action.values()))))>1e-10))


def make_driving_video(compiled,packets,trace,out,backend):
    import subprocess
    out=Path(out);frames=out/'video_frames';frames.mkdir()
    pairs=[dict(baseline=dict(ego_to_world=sample['ego_to_world']),new=packet) for sample,packet in zip(compiled['samples'],packets)]
    metrics=trajectory_metrics(pairs)
    (out/'pose_comparison.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
    for i,packet in enumerate(packets):
        sheet=Image.new('RGB',(1440,568),(20,20,20));draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'Executed ego trajectory | {backend} | iLQR + native Motion | left / front / right',fill='white')
        speed=trace[i-1]['state_after'][3] if i else compiled['execution_initial_speed_mps']
        draw.text((8,26),f"frame={packet['source_frame']} time={packet['timestamp_s']:.2f}s speed={speed:.2f}m/s | actual executed pose",fill='white')
        for col,name in enumerate(order):
            with Image.open(packet['cameras'][name]['rgb']) as source:
                im=source.convert('RGB');im.thumbnail((480,320));sheet.paste(im,(col*480,48))
        draw_trajectory_panel(draw,metrics,i,368,execution=True)
        sheet.save(frames/f'{i:05d}.png')
    video=out/'ego_driving_execution.mp4';ffmpeg=ffmpeg_path()
    subprocess.run([str(ffmpeg),'-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-n','-framerate',str(compiled['fps']),'-i',str(frames/'%05d.png'),
        '-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    return str(video),metrics


def execute_trajectory(bridge,args,request_id,compiled,plan):
    ffmpeg_path();vehicle=bridge.session.spec.vehicle;driver=TrackingDriver(plan,vehicle)
    version={k:args[k] for k in ('expected_generation','expected_revision')}
    initial_controls=bridge.control_count;out=bridge.output/'driving'/request_id;out.mkdir(parents=True,exist_ok=False)
    compiled=dict(compiled,execution_initial_speed_mps=plan['initial_speed_mps'])
    (out/'target_trajectory.json').write_text(json.dumps(compiled,indent=2,allow_nan=False))
    packets=[];trace=[];manifest=dict(completed=False,frames=packets,trace=trace,
        execution_backend=bridge.execution_backend,feedback_source='native_motion_state',
        perception_connected=False,uniad_inference_executed=False,collision_checked=False,
        controller='HUGSIM ILQRSolver at shared vehicle dt',motion_model='live_env.Motion')
    def save():
        (out/'execution.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    def state(index,status):
        from ego_attitude import pose_for_sample
        nominal=motion_pose(plan['anchor'],driver.motion)
        bridge.driving_state=dict(status=status,request_id=request_id,manifest=str(out/'execution.json'),
            generation=bridge.session.generation,edit_revision=bridge.session.revision,
            timestamp_s=compiled['samples'][index]['timestamp_s'],state=motion_state(driver.motion).tolist(),
            state_axes='forward,right,right_positive_heading,speed,right_positive_steering',
            attitude_policy=compiled.get('attitude_policy','initial_ego'),motion_ego_to_world=nominal.tolist(),
            ego_to_world=pose_for_sample(compiled,nominal,index).tolist())
    try:
        for i,sample in enumerate(compiled['samples']):
            if i:
                trace.append(driver.advance(i-1));bridge.control_count+=1
            state(i,'running')
            obs=dict(version,timestamp_s=sample['timestamp_s'],ego_to_world=bridge.driving_state['ego_to_world'],attitude_policy=compiled.get('attitude_policy','initial_ego'))
            packet=bridge.observe(obs,f'{request_id}_{i:04d}_executed',preview=False)
            packet['pose_source']='native Motion advanced by iLQR actions; JSON poses are tracking references only'
            packets.append(packet);save()
            print(f'Executed trajectory feedback {i}/{compiled["sample_count"]-1}: speed={driver.motion.speed:.3f}',flush=True)
        from live_env import Motion
        replay=Motion(dt=vehicle.dt,wheelbase=vehicle.wheelbase_m,start_speed=plan['initial_speed_mps']);replay.steer=plan['initial_steer_rad']
        replay_ok=True;pose_ok=np.allclose(packets[0]['ego_to_world'],compiled['samples'][0]['ego_to_world'],rtol=0,atol=1e-9)
        from ego_attitude import pose_for_sample,horizon_metrics
        for sample_index,(row,packet) in enumerate(zip(trace,packets[1:]),start=1):
            action=row['action'];validate_action(action,vehicle,replay.speed,replay.steer)
            replay_ok &= np.allclose(motion_state(replay),row['state_before'],rtol=0,atol=1e-9)
            replay.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
            replay_ok &= np.allclose(motion_state(replay),row['state_after'],rtol=0,atol=1e-9)
            pose_ok &= np.allclose(packet['ego_to_world'],pose_for_sample(compiled,motion_pose(plan['anchor'],replay),sample_index),rtol=0,atol=1e-9)
        manifest['attitude']=[horizon_metrics(packet,compiled['gravity_up_source']) for packet in packets]
        manifest['attitude_policy']=compiled.get('attitude_policy','initial_ego')
        video,metrics=make_driving_video(compiled,packets,trace,out,bridge.execution_backend)
        errors=[r['separation_m'] for r in metrics];heading=[abs(r['relative_yaw_degrees']) for r in metrics]
        audit=dict(all_actions_replay=bool(replay_ok),rendered_poses_from_motion=bool(pose_ok),
            controls_count=bridge.control_count-initial_controls==len(compiled['samples'])-1,
            observations_count=len(packets)==len(compiled['samples']),
            generation_preserved=bridge.session.generation==version['expected_generation'],
            revision_preserved=bridge.session.revision==version['expected_revision'],
            execution_seals_edits=bridge.session.editor.started,
            packets_match_time_and_version=all(packet['timestamp_s']==sample['timestamp_s'] and
                packet['source_frame']==sample['source_frame'] and packet['history_token']==version['expected_generation'] and
                packet['edit_revision']==version['expected_revision'] and packet['observation_history_committed']
                for packet,sample in zip(packets,compiled['samples'])))
        tracking=dict(max_position_error_m=max(errors),rms_position_error_m=float(np.sqrt(np.mean(np.square(errors)))),
            final_position_error_m=errors[-1],max_heading_error_degrees=max(heading),final_heading_error_degrees=heading[-1],
            position_tolerance_m=plan['max_tracking_error_m'],heading_tolerance_degrees=plan['max_tracking_heading_error_degrees'],
            passed=max(errors)<=plan['max_tracking_error_m'] and max(heading)<=plan['max_tracking_heading_error_degrees'])
        manifest.update(completed=all(audit.values()),audit=audit,tracking=tracking,video=video)
        if not manifest['completed']:raise RuntimeError('execution audit failed')
        bridge.execution_count+=1;state(len(packets)-1,'completed')
    except Exception as exc:
        if bridge.driving_state and bridge.driving_state.get('request_id')==request_id:
            bridge.driving_state['status']='failed'
        manifest['error']=str(exc)
        raise
    finally:save()
    return dict(manifest=str(out/'execution.json'),video=video,audit=audit,tracking=tracking,
        executed_controls=len(trace),rendered_packets=len(packets),preview_only=False,
        control_executed=True,closed_loop_driving=True,feedback_source='native_motion_state',
        perception_connected=False,uniad_inference_executed=False,collision_checked=False,
        initial_state=[0.,0.,0.,plan['initial_speed_mps'],plan['initial_steer_rad']],
        final_state=motion_state(driver.motion).tolist(),generation=bridge.session.generation,edit_revision=bridge.session.revision)
