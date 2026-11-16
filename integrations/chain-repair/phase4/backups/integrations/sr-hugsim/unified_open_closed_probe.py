"""One frozen-policy edited Waymo172 open/closed-loop comparison; no N1 grant."""

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
import hashlib
import os
pass  # Path is provided by the portable boundary above
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np
from scene_api import BASE, SceneSpec, SceneSession, validate_action
from waymo_adapter import SR_TO_HUGSIM as T
from uniad_bridge import to_uniad, placement_reference

ADAPTER = BASE/'waymo23-native-import'
sys.path[:0] = [str(ADAPTER), str(ADAPTER/'waymo172_closed_loop_v1')]
from closed172_common import (atomic_json, atomic_pickle, wait_json, native_delta,
    optimize, emergency_stop_action, reference_check, integrate_state)
from build_source_sequence import training_lidar_mount, recorded_kinematics
from frozen_bev_experiment import fingerprint

SEED = 20261021
CAMERA = 'CAM_FRONT_RIGHT'


from comparison_video import compose, encode, compose_three_closed, THREE_CAMERAS
from planning_guard import ObstacleMemory, RecoveryController, POLICY


def run(request_path, output):
    request = _hugsim_expand(json.loads(Path(request_path).read_text()))
    extended = request.get('schema') == 'unified_three_closed_hold_v1'
    allowed = {'schema', 'scene_manifest', 'insert_request', 'start_frame', 'steps'}
    if extended: allowed |= {'time_policy'}
    allowed |= {'resume_checkpoint'}
    if set(request)-allowed or allowed-{'resume_checkpoint'}-set(request) or request['schema'] not in ('unified_open_closed_172_v1','unified_three_closed_hold_v1'):
        raise ValueError('Unexpected comparison request')
    if request.get('resume_checkpoint') and not extended:raise ValueError('Resume requires the three-view extended protocol')
    start, steps = request['start_frame'], request['steps']
    if extended:
        if request['time_policy'] != 'hold_last_source_frame' or isinstance(start,bool) or not isinstance(start,int) or start<0 or isinstance(steps,bool) or not isinstance(steps,int) or not 51 <= steps <= 200:
            raise ValueError('Invalid explicit hold-last-source diagnostic request')
    elif start != 146 or steps != 50: raise ValueError('This preview uses frames146..196, 50 feedback actions')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'comparison_frames').mkdir(); (output/'open_loop').mkdir(); (output/'closed_loop').mkdir()
    worker_folder = output/'uniad_worker'; worker_folder.mkdir(); (worker_folder/'step_000').mkdir()
    manifest = _hugsim_expand(json.loads(Path(request['scene_manifest']).read_text())); manifest['render_scale'] = 1.
    atomic_json(output/'scene_manifest.json', manifest)
    spec = SceneSpec.read(output/'scene_manifest.json')
    if (not extended and str(spec.source['scene']) != '172') or (start >= spec.frames or (not extended and start+steps >= spec.frames)) or not np.isclose(1/spec.fps, spec.vehicle.dt):
        raise ValueError('Scene/clock/window mismatch')
    insert = _hugsim_expand(json.loads(Path(request['insert_request']).read_text())); atomic_json(output/'insert_request.json', insert)
    calibration = _hugsim_expand(json.loads(Path(insert['asset_calibration']).read_text()))
    reference = placement_reference(insert, calibration); atomic_json(output/'asset_reference.json', reference)
    inputs = Path(spec.source['inputs']); raw_start = spec.source['start_timestep']
    poses = np.stack([np.loadtxt(inputs/'ego_pose'/f'{raw_start+i:03d}.txt') for i in range(spec.frames)])
    relative = np.linalg.inv(poses[0])[None] @ poses
    mount = training_lidar_mount(BASE/'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    vehicle = spec.vehicle
    replan_period=float(_hugsim_expand(json.loads((ADAPTER/'configs/planning.json').read_text())).get('global_replan_interval_s',vehicle.dt))
    replan_ticks=int(round(replan_period/vehicle.dt))
    if replan_ticks<1 or not np.isclose(replan_ticks*vehicle.dt,replan_period):raise ValueError('Global replanning period must match control clock')
    source_end_time=(spec.frames-1)/spec.fps
    initial_native = T @ relative[start] @ T.T
    speed, steer, _ = recorded_kinematics(relative, start, spec.fps, vehicle.wheelbase_m)
    checkpoint = BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    scene=spec.source['scene']
    report = dict(schema='unified_open_closed_comparison_v1', status='running', scene=scene,
        start_frame=start, steps_requested=steps, dt=vehicle.dt, camera=CAMERA,
        training=False, n1_validation_suspended=True, formal_evaluation_admitted=False,
        closed_loop_run=True, control_actions_executed=False,
        open_loop='recorded ego poses; edited scene',
        closed_loop='frozen UniAD current planning BEV, history perception BEV; evidence-tiered rectangular MPC; native Motion.advance',
        guard_policy=POLICY,
        ground_truth_used_by_optimizer=False, recorded_future_ego_used_for_closed_motion=False,
        route_commands_from_recorded_route=True, trace=[], observations=[])
    audit_files=[ADAPTER/'configs/planning.json',Path(__file__),Path(__file__).with_name('comparison_video.py'),Path(__file__).with_name('trajectory_baseline_score.py'),Path(__file__).with_name('native_geometry_evaluation.py'),Path(__file__).with_name('closed_loop_resume.py'),ADAPTER/'planning_guard.py',ADAPTER/'depth_sensor_map.py',ADAPTER/'visibility_curtain.py',ADAPTER/'depth_contract.py',BASE/'code/HUGSIM/gaussian_renderer/__init__.py',BASE/'code/HUGSIM/gaussian_renderer/surface_sensor.py',ADAPTER/'live_env.py',Path(__file__).with_name('native_scene_backend.py'),Path(__file__).with_name('scene_api.py'),Path(__file__).with_name('uniad_bridge.py'),ADAPTER/'depth_surface_geometry.py',ADAPTER/'pseudo_lidar_adapter.py',ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py',BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py',ADAPTER/'waymo172_closed_loop_v1/closed172_common.py',ADAPTER/'runtime_configuration.py',Path(manifest['vehicle_config'])]
    if spec.source.get('planning_ground_skin'):
        audit_files.extend([spec.source_report,Path(spec.source['planning_ground_skin']['data']),Path(spec.source['planning_ground_skin']['metadata'])])
    implementation=lambda:{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in audit_files}
    report.update(time_policy='hold_last_source_frame' if extended else 'source_time_only',
        source_end_timestamp_s=source_end_time,three_view=extended,
        camera_order=list(THREE_CAMERAS) if extended else [CAMERA],
        requested_duration_seconds=steps*vehicle.dt)
    if extended: report['open_loop']='not rendered; closed-loop-only three-camera diagnostic'
    for name in THREE_CAMERAS if extended else []: (output/'closed_loop'/name).mkdir()
    identity=dict(manifest=manifest,source=spec.source,insert=insert,start=start,steps=steps,time_policy=request.get('time_policy'),source_checkpoint_stat=[Path(spec.source['checkpoint']).stat().st_size,Path(spec.source['checkpoint']).stat().st_mtime_ns],asset_stat=[(Path(insert['asset'])/'gs.pth').stat().st_size,(Path(insert['asset'])/'gs.pth').stat().st_mtime_ns])
    report['experiment_signature']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    from closed_loop_resume import save_checkpoint,load_checkpoint,copy_frames,replay_model_history
    resume=load_checkpoint(request['resume_checkpoint'],implementation(),BASE,report['experiment_signature']) if request.get('resume_checkpoint') else None
    start_index=int(resume['next_index']) if resume else 0
    if resume:
        report=resume['report'];report['status']='running';report.pop('error',None)
        report.setdefault('resumes',[]).append(dict(checkpoint=request['resume_checkpoint'],next_index=start_index))
        copy_frames(resume,output,THREE_CAMERAS)
    report['implementation_before']=implementation()
    began = time.monotonic()
    atomic_json(output/'progress.json', dict(stage='hashing_frozen_checkpoint'))
    report['checkpoint_before'] = fingerprint(checkpoint)
    token = resume['token'] if resume else str(uuid4()); process = None; log = None; min_gap=float('inf')
    os.environ['BASE'] = str(BASE)
    os.environ['MPLCONFIGDIR'] = str(BASE/'outputs/gpu_test_service/matplotlib_cache')
    try:
        with SceneSession() as session:
            atomic_json(output/'progress.json', dict(stage='loading_edited_scene'))
            session.load_scene(output/'scene_manifest.json'); session.edit(insert)
            from native_geometry_evaluation import NativeVehicleEvaluation
            native_evaluation=NativeVehicleEvaluation(session.backend.env.models['RigidNodes'])
            from live_env import Motion
            motion = Motion(dt=vehicle.dt, wheelbase=vehicle.wheelbase_m, start_speed=speed)
            motion.steer = steer; motion.timestamp = start/spec.fps
            if resume:motion.__dict__.update(resume['motion'])
            if reference_check(initial_native, reference, vehicle)['overlap']:
                raise RuntimeError('Initial ego overlaps inserted reference')
            first = session.render(relative[start], start/spec.fps)
            def model_packet(packet):
                # Recorded route is used only for navigation command, never the motion update.
                nearest = int(np.argmin(np.linalg.norm(relative[:,:2,3]-np.asarray(packet['ego_to_world'])[:2,3],axis=1)))
                _, _, command = recorded_kinematics(relative, nearest, spec.fps, vehicle.wheelbase_m)
                obs, info = to_uniad(packet, mount, speed_mps=motion.speed, steering_rad=motion.steer,
                    command=command, scene_token=token, dt=vehicle.dt,
                    acceleration_mps2=motion.last_accel, steering_rate_radps=motion.last_rate)
                if packet.get('background_frozen'):
                    info['declared_static_epoch']=token+':'+str(packet['edit_revision'])+':'+str(packet['background_timestamp_s'])
                info.update(closed_loop_run=True, control_actions_executed=True,
                    ego_kinematics_source='native_motion_control_feedback')
                return obs, info
            atomic_pickle(worker_folder/'step_000/observation.pkl', model_packet(first))
            env = dict(os.environ, BASE=str(BASE), PYTHONUNBUFFERED='1', BEV_EXPERIMENT_SEED=str(SEED),
                PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', UNIAD_INIT_TRACE='1',
                UNIAD_STARTUP_REPORT=str(worker_folder/'startup.json'))
            for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)
            log = (output/'uniad_worker.log').open('w')
            process = subprocess.Popen([str(BASE/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
                str(ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py'), '--job', str(worker_folder),
                '--planning-bev', 'current', '--model-profile', 'calibration_geometry_v1', '--diagnostic-closed-loop'],
                env=env, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output/'progress.json', dict(stage='loading_resident_uniad'))
            report['worker_ready'] = wait_json(worker_folder/'ready.json', process, timeout=1200.)
            if report['worker_ready']['planning_bev'] != 'current': raise ValueError('Wrong planning profile')
            from PIL import Image
            from depth_sensor_map import DepthObstacleMemory
            depth_memory=DepthObstacleMemory(vehicle)
            warm = None; controller = RecoveryController(); obstacle_memory = ObstacleMemory.from_source(spec.source); mode = 'initial state'; min_gap = float('inf')
            if resume:
                warm=resume['warm'];controller=resume['controller'];obstacle_memory=resume['obstacle_memory'];depth_memory=resume['depth_memory']
                atomic_json(output/'progress.json',dict(stage='restoring_model_history',frames=start_index))
                replay_model_history(resume,worker_folder,process,wait_json,atomic_json,atomic_pickle)
                min_gap=resume['minimum_gap']
            for index in range(start_index,steps+1):
                frame = start+index; timestamp = frame/spec.fps
                source_frame=min(frame,spec.frames-1) if extended else frame
                atomic_json(output/'progress.json', dict(stage='closed_loop_feedback', step=index, total_steps=steps,
                    source_frame=source_frame,simulation_frame=frame))
                native = initial_native @ native_delta(motion.x, motion.z, motion.yaw)
                closed_pose = relative[start] if index == 0 else T.T @ native @ T
                background_time=source_frame/spec.fps
                closed_packet = first if index == 0 else session.render(closed_pose, background_time)
                # Render clock can hold; model/ego clock must remain continuous.
                closed_packet=dict(closed_packet,timestamp_s=timestamp,
                    background_timestamp_s=background_time,background_frozen=timestamp>source_end_time+1e-8)
                right = Image.fromarray(closed_packet['cameras'][CAMERA]['rgb'])
                ospeed, _, _ = recorded_kinematics(relative, source_frame, spec.fps, vehicle.wheelbase_m)
                if extended:
                    for name in THREE_CAMERAS:
                        Image.fromarray(closed_packet['cameras'][name]['rgb']).save(output/f'closed_loop/{name}/{index:03d}.png')
                    note=f'BACKGROUND HELD at {source_end_time:.1f}s' if closed_packet['background_frozen'] else 'Original background time'
                    compose_three_closed(closed_packet,index*vehicle.dt,motion.speed,mode,note).save(output/f'comparison_frames/{index:03d}.png')
                else:
                    open_packet=first if index == 0 else session.render(relative[frame],timestamp)
                    left=Image.fromarray(open_packet['cameras'][CAMERA]['rgb'])
                    left.save(output/f'open_loop/{index:03d}.png');right.save(output/f'closed_loop/{index:03d}.png')
                    compose(left,right,index*vehicle.dt,ospeed,motion.speed,mode).save(output/f'comparison_frames/{index:03d}.png')
                native_check=native_evaluation.check(closed_pose,source_frame,vehicle)
                native_gap=native_check['nearest']['gap_m']
                if native_gap is not None:report['minimum_native_reference_gap_m']=min(report.get('minimum_native_reference_gap_m',float('inf')),native_gap)
                if native_check['overlap']:raise RuntimeError('Post-hoc native-vehicle reference overlap; diagnostic stopped')
                gap = reference_check(native,reference,vehicle); min_gap = min(min_gap,gap['gap_m'])
                if gap['overlap']: raise RuntimeError('Post-hoc inserted-vehicle footprint overlap; diagnostic stopped')
                report['observations'].append(dict(index=index, timestamp=timestamp, source_frame=source_frame,simulation_frame=frame,
                    background_timestamp_s=background_time,background_frozen=closed_packet['background_frozen'],
                    open_ego_to_world=relative[source_frame].tolist(),closed_ego_to_world=np.asarray(closed_pose).tolist(),
                    open_speed_mps=ospeed,closed_speed_mps=motion.speed,control_mode_from_previous_step=mode,
                    inserted_reference_gap_m=gap['gap_m'],native_vehicle_reference=native_check))
                atomic_json(output/'run_report.json',report)
                if index == steps: break
                folder = worker_folder/f'step_{index:03d}'; folder.mkdir(exist_ok=True)
                packet_path = folder/'observation.pkl'; obs,info = model_packet(closed_packet)
                if index: atomic_pickle(packet_path,(obs,info))
                req = dict(index=index,packet=str(packet_path))
                if index == 0: req['diagnostic_seed'] = SEED
                atomic_json(worker_folder/f'request_{index:03d}.json',req)
                response = wait_json(worker_folder/f'done_{index:03d}.json',process,timeout=300.)
                if not np.isclose(response['timestamp'],timestamp): raise ValueError('Inference clock mismatch')
                with np.load(response['perception']) as result:
                    plan,occupancy = result['trajectory'].copy(),result['occupancy'].copy()
                    candidates = [result['candidate_'+key].copy() for key in ('boxes','scores','labels')]
                    if 'geometry_candidate_boxes' in result:
                        candidates[0] = result['geometry_candidate_boxes'].copy()
                reliability = _hugsim_expand(json.loads(Path(response['reliability']).read_text()))
                reliability = obstacle_memory.update(reliability, *candidates, info, current_occupancy=np.asarray(occupancy).reshape(5,200,200)[0])
                depth_points,depth_audit=depth_memory.update(obs,info)
                reliability['unknown_visibility_obstacles_xy']=depth_audit.pop('unknown_visibility_points_xy',[])
                reliability['unknown_visibility_geometry']=depth_audit.get('unknown_visibility_geometry',{})
                reliability['observed_depth_obstacles_xy']=depth_points.tolist()
                reliability['observed_depth_geometry']=depth_audit
                atomic_json(folder/'depth_geometry.json',depth_audit)
                atomic_json(folder/'guard_memory_report.json', reliability)
                recovery_cap=controller.speed_cap(reliability)
                if recovery_cap is not None:
                    previous_cap=reliability.get('speed_limit_mps')
                    reliability=dict(reliability,speed_limit_mps=min(recovery_cap,previous_cap) if previous_cap is not None else recovery_cap)
                try: optimized = optimize(plan,occupancy,info,warm=warm,reliability=reliability,vehicle=vehicle,_reuse_verified_warm=bool(index%replan_ticks))
                except ValueError as exc:
                    if not str(exc).startswith('Empty occupancy:'): raise
                    optimized = dict(selected=None,reason=str(exc),ground_truth_used_by_optimizer=False)
                atomic_json(folder/'optimization.json',optimized)
                action,decision=controller.update(optimized,motion.speed,timestamp,vehicle.dt,vehicle.emergency_braking_mps2)
                atomic_json(folder/'control_decision.json',decision)
                selected=optimized['selected']
                following=decision['state_after'] in ('normal','caution','recovering')
                warm=np.asarray(selected['controls']) if selected is not None and following else None
                mode=(decision['state_after']+' / MPC') if following else (decision['state_after']+' / reassessing')
                validate_action(dict(acc_mps2=action['acc'],steer_rate_radps=action['steer_rate']),vehicle,motion.speed,motion.steer)
                before = np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer])
                report['controls_attempted']=index+1
                motion.advance(action)
                after = np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer])
                expected = integrate_state(before,[action['acc'],action['steer_rate']],vehicle.dt,vehicle.wheelbase_m)
                np.testing.assert_allclose(after,expected,rtol=0,atol=1e-10)
                for u in np.linspace(0,1,11):
                    p = before*(1-u)+after*u
                    check = reference_check(initial_native@native_delta(*p[:3]),reference,vehicle)
                    min_gap = min(min_gap,check['gap_m'])
                    if check['overlap']: raise RuntimeError('Interpolated reference overlap after control')
                    checked_native=initial_native@native_delta(*p[:3])
                    checked_source=T.T@checked_native@T
                    native_result=native_evaluation.check(checked_source,source_frame,vehicle)
                    native_gap=native_result['nearest']['gap_m']
                    if native_gap is not None:report['minimum_native_reference_gap_m']=min(report.get('minimum_native_reference_gap_m',float('inf')),native_gap)
                    if native_result['overlap']:
                        raise RuntimeError('Interpolated native-vehicle reference overlap after control')
                temporal = _hugsim_expand(json.loads((folder/'temporal_state.json').read_text()))
                if temporal['history_bev_present'] != (index > 0) or not temporal['metadata_cache_is_tracking_cache']:
                    raise RuntimeError('Unexpected temporal BEV state')
                report['trace'].append(dict(step=index,timestamp=timestamp,action=action,mode=mode,
                    optimization_reason=optimized.get('reason'),control_decision=decision,state_before=before.tolist(),state_after=after.tolist(),
                    neural_plan=plan.tolist(),perception=response['perception'],reliability=response['reliability'],
                    temporal=temporal,physics_integration_verified=True))
                report['control_actions_executed'] = True
                atomic_json(output/'run_report.json',report)
                if obs.get('depth_alpha'):np.savez_compressed(folder/'observed_depth_alpha.npz',**obs['depth_alpha'])
                if obs.get('depth_variance'):np.savez_compressed(folder/'observed_depth_variance.npz',**obs['depth_variance'])
                for field in ['surface_depth','surface_valid']:
                    if obs.get(field):np.savez_compressed(folder/(field+'.npz'),**obs[field])
                atomic_pickle(folder/'sensor_metadata.pkl',info)
                np.savez_compressed(folder/'observed_depth.npz',**obs['depth'])
                atomic_json(folder/'sensor_metadata.json',_hugsim_expand(json.loads(json.dumps(info,default=lambda value:str(value) if isinstance(value,Path) else np.asarray(value).tolist()))))
                if index: packet_path.unlink()
                if (index+1)%10==0:
                    save_checkpoint(output/'driver_checkpoint.pkl',implementation=implementation(),source_result=output,next_index=index+1,token=token,motion=motion,warm=warm,controller=controller,obstacle_memory=obstacle_memory,depth_memory=depth_memory,report=report,minimum_gap=min_gap)
                    atomic_json(output/'resume_available.json',dict(checkpoint=str(output/'driver_checkpoint.pkl'),next_index=index+1))
                print(f'Completed feedback {index+1}/{steps}: {mode}; speed={motion.speed:.3f}',flush=True)
            report['minimum_inserted_reference_gap_m'] = min_gap
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        atomic_json(worker_folder/'stop.json',{})
        if process is not None:
            try: process.wait(timeout=20.)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=10.)
                except subprocess.TimeoutExpired: process.kill();process.wait()
        if log is not None: log.close()
        report['implementation_after']=implementation()
        report['implementation_unchanged']=report['implementation_after']==report['implementation_before']
        report['checkpoint_after'] = fingerprint(checkpoint)
        report['checkpoint_unchanged'] = report['checkpoint_before'] == report['checkpoint_after']
        if np.isfinite(min_gap): report['minimum_inserted_reference_gap_m']=min_gap
        if extended:
            from trajectory_baseline_score import score_report
            report['trajectory_score']=score_report(report,reference,vehicle)
            atomic_json(output/'trajectory_score.json',report['trajectory_score'])
            if report['status']=='failed':
                count=len(list((output/'comparison_frames').glob('*.png')))
                if count:
                    try:
                        report['video']=encode(output,spec.fps,count,filename=f'{scene}_edited_closed_three_views_partial.mp4')
                        report['video_partial']=True
                    except Exception as exc: report['video_error']=repr(exc)
        atomic_json(output/'run_report.json',report)
    if not report['implementation_unchanged']: raise RuntimeError('Control implementation changed during run')
    if not report['checkpoint_unchanged']: raise RuntimeError('Frozen checkpoint changed')
    report['video'] = encode(output,spec.fps,steps+1,filename=f'{scene}_edited_closed_three_views_{steps*vehicle.dt:g}s.mp4') if extended else encode(output,spec.fps,steps+1)
    report['checks'] = dict(first_images_identical=bool(np.array_equal(
        np.asarray(Image.open(output/'closed_loop/CAM_FRONT_RIGHT/000.png')) if extended else np.asarray(Image.open(output/'open_loop/000.png')),
        np.asarray(Image.open(output/'closed_loop/CAM_FRONT_RIGHT/000.png')) if extended else np.asarray(Image.open(output/'closed_loop/000.png')))),
        first_poses_equal=bool(np.allclose(report['observations'][0]['open_ego_to_world'],report['observations'][0]['closed_ego_to_world'])),
        action_count=len(report['trace']),observation_count=len(report['observations']),
        all_actions_integrated=all(row['physics_integration_verified'] for row in report['trace']),
        history_cache_checked_frames=len(report['trace']),video_full_decode_passed=True)
    if extended:
        report['checks'].pop('first_images_identical')
        report['checks']['three_camera_frame_counts']={name:len(list((output/'closed_loop'/name).glob('*.png'))) for name in THREE_CAMERAS}
        report['checks']['continuous_model_clock']=all(np.isclose(row['timestamp'],(start+i)*vehicle.dt) for i,row in enumerate(report['trace']))
        if any(count != steps+1 for count in report['checks']['three_camera_frame_counts'].values()) or not report['checks']['continuous_model_clock']:
            raise RuntimeError('Extended camera or model clock mismatch')
    if (not extended and not report['checks']['first_images_identical']) or not report['checks']['first_poses_equal']:
        raise RuntimeError('Comparison start mismatch')
    last = report['observations'][-1]
    report['final_open_closed_position_difference_m'] = float(np.linalg.norm(
        np.asarray(last['open_ego_to_world'])[:3,3]-np.asarray(last['closed_ego_to_world'])[:3,3]))
    report.update(status='completed',wall_seconds=time.monotonic()-began)
    if extended:
        from trajectory_baseline_score import score_report
        report['trajectory_score']=score_report(report,reference,vehicle)
        atomic_json(output/'trajectory_score.json',report['trajectory_score'])
    atomic_json(output/'run_report.json',report)
    atomic_json(output/'progress.json',dict(stage='completed',video=report['video']['path']))
    return report
