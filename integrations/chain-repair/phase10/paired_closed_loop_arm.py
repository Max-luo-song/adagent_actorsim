"""Independent frozen-policy dynamic closed-loop arm; no N1 grant."""

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
    dynamic = request.get('schema') == 'unified_three_closed_dynamic_v1'
    extended = request.get('schema') in ('unified_three_closed_hold_v1','unified_three_closed_dynamic_v1')
    allowed = {'schema', 'scene_manifest', 'insert_request', 'start_frame', 'steps'}
    if extended: allowed |= {'time_policy'}
    allowed |= {'resume_checkpoint', 'arm', 'stop_on_stationary', 'complete_on_pass'}
    if set(request)-allowed or allowed-{'resume_checkpoint','arm','stop_on_stationary','complete_on_pass'}-set(request) or request['schema'] not in ('unified_open_closed_172_v1','unified_three_closed_hold_v1','unified_three_closed_dynamic_v1'):
        raise ValueError('Unexpected comparison request')
    if request.get('resume_checkpoint') and not extended:raise ValueError('Resume requires the three-view extended protocol')
    arm=request.get('arm','edited')
    if arm not in ('edited','unedited'):raise ValueError('Invalid closed-loop arm')
    if not dynamic:raise ValueError('Paired arms require source-time dynamic protocol')
    if request.get('resume_checkpoint'):raise ValueError('Paired arms must start with fresh independent histories')
    for key in ('stop_on_stationary','complete_on_pass'):
        if key in request and type(request[key]) is not bool:raise ValueError('Invalid boolean '+key)
    edited=arm=='edited'
    stop_on_stationary=request.get('stop_on_stationary',edited)
    complete_on_pass=request.get('complete_on_pass',edited)
    start, steps = request['start_frame'], request['steps']
    if extended:
        if request['time_policy'] != ('source_time_only' if dynamic else 'hold_last_source_frame') or isinstance(start,bool) or not isinstance(start,int) or start<0 or isinstance(steps,bool) or not isinstance(steps,int) or not (1 if dynamic else 51) <= steps <= 200:
            raise ValueError('Invalid explicit hold-last-source diagnostic request')
    elif start != 146 or steps != 50: raise ValueError('This preview uses frames146..196, 50 feedback actions')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'comparison_frames').mkdir(); (output/'open_loop').mkdir(); (output/'closed_loop').mkdir()
    worker_folder = output/'uniad_worker'; worker_folder.mkdir(); (worker_folder/'step_000').mkdir()
    manifest = _hugsim_expand(json.loads(Path(request['scene_manifest']).read_text())); manifest['render_scale'] = 1.
    atomic_json(output/'scene_manifest.json', manifest)
    spec = SceneSpec.read(output/'scene_manifest.json')
    if (not extended and str(spec.source['scene']) != '172') or (start >= spec.frames or ((not extended or dynamic) and start+steps >= spec.frames)) or not np.isclose(1/spec.fps, spec.vehicle.dt):
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
        route_commands_from_recorded_route=True, arm=arm, stationary_threshold_mps=0.001, trace=[], observations=[])
    audit_files=[ADAPTER/'configs/planning.json',Path(__file__),(BASE/'integrations/sr-hugsim').joinpath('comparison_video.py'),(BASE/'integrations/sr-hugsim').joinpath('trajectory_baseline_score.py'),(BASE/'integrations/sr-hugsim').joinpath('native_geometry_evaluation.py'),(BASE/'integrations/sr-hugsim').joinpath('native_actor_evaluation.py'),(BASE/'integrations/sr-hugsim').joinpath('closed_loop_resume.py'),(BASE/'integrations/sr-hugsim').joinpath('dynamic_completion.py'),ADAPTER/'planning_guard.py',ADAPTER/'depth_sensor_map.py',ADAPTER/'visibility_curtain.py',ADAPTER/'depth_contract.py',ADAPTER/'drivable_area.py',ADAPTER/'road_route_seed.py',BASE/'code/HUGSIM/gaussian_renderer/__init__.py',BASE/'code/HUGSIM/gaussian_renderer/surface_sensor.py',ADAPTER/'live_env.py',(BASE/'integrations/sr-hugsim').joinpath('native_scene_backend.py'),(BASE/'integrations/sr-hugsim').joinpath('scene_api.py'),(BASE/'integrations/sr-hugsim').joinpath('uniad_bridge.py'),ADAPTER/'depth_surface_geometry.py',ADAPTER/'pseudo_lidar_adapter.py',ADAPTER/'waymo172_closed_loop_v1/closed172_worker.py',BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py',ADAPTER/'waymo172_closed_loop_v1/closed172_common.py',ADAPTER/'runtime_configuration.py',Path(manifest['vehicle_config'])]
    audit_files.extend([Path(__file__).with_name(name) for name in ('drivable_area.py','road_route_seed.py','road_prediction_memory.py','closed172_common.py','stop_diagnosis.py','bounded_visibility_geometry.py','navigation_anchor.py','committed_route.py')])
    if spec.source.get('planning_ground_skin'):
        audit_files.extend([spec.source_report,Path(spec.source['planning_ground_skin']['data']),Path(spec.source['planning_ground_skin']['metadata'])])
    implementation=lambda:{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in audit_files}
    report.update(time_policy='hold_last_source_frame' if extended and not dynamic else 'source_time_only',
        source_end_timestamp_s=source_end_time,three_view=extended,
        camera_order=list(THREE_CAMERAS) if extended else [CAMERA],
        requested_duration_seconds=steps*vehicle.dt)
    if extended: report['open_loop']='not rendered; independent '+arm+' closed-loop three-camera diagnostic'
    if dynamic:
        report['termination_policy']='pass asset fully with required clearance, then 0.5s continued motion; otherwise source window exhausted'
        report['avoidance_success']=False
        report['termination_reason']='source_window_exhausted'
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
            session.load_scene(output/'scene_manifest.json')
            if edited:session.edit(insert)
            from native_geometry_evaluation import NativeVehicleEvaluation
            native_evaluation=NativeVehicleEvaluation(session.backend.env.models['RigidNodes'])
            from native_actor_evaluation import NativeActorEvaluation
            actor_evaluation=NativeActorEvaluation({name:session.backend.env.models[name] for name in ('RigidNodes','DeformableNodes','SMPLNodes') if name in session.backend.env.models})
            report['actor_reference_groups']=list(actor_evaluation.groups)
            from live_env import Motion
            motion = Motion(dt=vehicle.dt, wheelbase=vehicle.wheelbase_m, start_speed=speed)
            motion.steer = steer; motion.timestamp = start/spec.fps
            if resume:motion.__dict__.update(resume['motion'])
            if edited and reference_check(initial_native, reference, vehicle)['overlap']:
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
            from road_prediction_memory import RoadPredictionMemory
            road_memory=RoadPredictionMemory(vehicle)
            from navigation_anchor import NavigationAnchor
            navigation_anchor=NavigationAnchor(vehicle.half_length_m)
            from committed_route import CommittedRoute
            committed_route=CommittedRoute()
            warm = None; controller = RecoveryController(); obstacle_memory = ObstacleMemory.from_source(spec.source); mode = 'initial state'; min_gap = float('inf')
            if resume:
                warm=resume['warm'];controller=resume['controller'];obstacle_memory=resume['obstacle_memory'];depth_memory=resume['depth_memory']
                atomic_json(output/'progress.json',dict(stage='restoring_model_history',frames=start_index))
                replay_model_history(resume,worker_folder,process,wait_json,atomic_json,atomic_pickle)
                min_gap=resume['minimum_gap']
            for index in range(start_index,steps+1):
                frame = start+index; timestamp = frame/spec.fps
                source_frame=min(frame,spec.frames-1) if extended and not dynamic else frame
                atomic_json(output/'progress.json', dict(stage='closed_loop_feedback', step=index, total_steps=steps,
                    source_frame=source_frame,simulation_frame=frame))
                native = initial_native @ native_delta(motion.x, motion.z, motion.yaw)
                closed_pose = relative[start] if index == 0 else T.T @ native @ T
                background_time=source_frame/spec.fps
                closed_packet = first if index == 0 else session.render(closed_pose, background_time)
                # Render clock can hold; model/ego clock must remain continuous.
                closed_packet=dict(closed_packet,timestamp_s=timestamp,
                    background_timestamp_s=background_time,background_frozen=timestamp>source_end_time+1e-8)
                obs,info=model_packet(closed_packet)
                right = Image.fromarray(closed_packet['cameras'][CAMERA]['rgb'])
                ospeed, _, _ = recorded_kinematics(relative, source_frame, spec.fps, vehicle.wheelbase_m)
                if extended:
                    for name in THREE_CAMERAS:
                        Image.fromarray(closed_packet['cameras'][name]['rgb']).save(output/f'closed_loop/{name}/{index:03d}.png')
                    note=f'BACKGROUND HELD at {source_end_time:.1f}s' if closed_packet['background_frozen'] else arm.upper()+' | live source time'
                    compose_three_closed(closed_packet,index*vehicle.dt,motion.speed,mode,note).save(output/f'comparison_frames/{index:03d}.png')
                else:
                    open_packet=first if index == 0 else session.render(relative[frame],timestamp)
                    left=Image.fromarray(open_packet['cameras'][CAMERA]['rgb'])
                    left.save(output/f'open_loop/{index:03d}.png');right.save(output/f'closed_loop/{index:03d}.png')
                    compose(left,right,index*vehicle.dt,ospeed,motion.speed,mode).save(output/f'comparison_frames/{index:03d}.png')
                native_check=native_evaluation.check(closed_pose,source_frame,vehicle)
                native_gap=native_check['nearest']['gap_m']
                if native_gap is not None:report['minimum_native_reference_gap_m']=min(report.get('minimum_native_reference_gap_m',float('inf')),native_gap)
                actor_check=actor_evaluation.check(closed_pose,source_frame,vehicle)
                actor_gap=actor_check['nearest']['gap_m']
                if actor_gap is not None:report['minimum_all_actor_reference_gap_m']=min(report.get('minimum_all_actor_reference_gap_m',float('inf')),actor_gap)
                if actor_check['overlap']:
                    report['terminal_actor_reference']=dict(actor_check,source_frame=float(source_frame),ego_to_world=np.asarray(closed_pose).tolist())
                    raise RuntimeError('Post-hoc actor reference overlap; diagnostic stopped')
                if native_check['overlap']:raise RuntimeError('Post-hoc native-vehicle reference overlap; diagnostic stopped')
                gap = reference_check(native,reference,vehicle) if edited else dict(overlap=False,gap_m=None)
                if edited:min_gap = min(min_gap,gap['gap_m'])
                if gap['overlap']: raise RuntimeError('Post-hoc inserted-vehicle footprint overlap; diagnostic stopped')
                report['observations'].append(dict(index=index, timestamp=timestamp, source_frame=source_frame,simulation_frame=frame,
                    background_timestamp_s=background_time,background_frozen=closed_packet['background_frozen'],
                    open_ego_to_world=relative[source_frame].tolist(),closed_ego_to_world=np.asarray(closed_pose).tolist(),
                    open_speed_mps=ospeed,closed_speed_mps=motion.speed,control_mode_from_previous_step=mode,
                    inserted_reference_gap_m=gap['gap_m'],native_vehicle_reference=native_check,all_actor_reference=actor_check,
                    navigation_rejoin=navigation_anchor.errors(info)))
                atomic_json(output/'run_report.json',report)
                if stop_on_stationary and index>0 and motion.speed<=report['stationary_threshold_mps']:
                    from stop_diagnosis import diagnose_stop
                    report.update(termination_reason='edited_ego_stopped' if edited else 'unedited_ego_stopped',
                        stopped_at_elapsed_s=index*vehicle.dt,stop_diagnosis=diagnose_stop(report,output))
                    atomic_json(output/'stop_diagnosis.json',report['stop_diagnosis'])
                    atomic_json(output/'run_report.json',report)
                    break
                if dynamic and edited:
                    # Evaluation controls termination only. Reference geometry
                    # remains absent from all perception/planner inputs.
                    from dynamic_completion import evaluate_completion
                    from navigation_anchor import rejoin_completion
                    completion=rejoin_completion(evaluate_completion(report,reference,vehicle),report['observations'],vehicle)
                    report['dynamic_completion']=completion
                    if completion['success'] and complete_on_pass:
                        report.update(avoidance_success=True,termination_reason='passed_and_continued_safely')
                        break
                if index == steps: break
                folder = worker_folder/f'step_{index:03d}'; folder.mkdir(exist_ok=True)
                packet_path = folder/'observation.pkl'
                if index: atomic_pickle(packet_path,(obs,info))
                req = dict(index=index,packet=str(packet_path))
                if index == 0: req['diagnostic_seed'] = SEED
                atomic_json(worker_folder/f'request_{index:03d}.json',req)
                response = wait_json(worker_folder/f'done_{index:03d}.json',process,timeout=300.)
                if not np.isclose(response['timestamp'],timestamp): raise ValueError('Inference clock mismatch')
                with np.load(response['perception']) as result:
                    from drivable_area import from_saved,DrivableArea
                    road_contract=from_saved(result) if dynamic else None
                    plan,occupancy = result['trajectory'].copy(),result['occupancy'].copy()
                    candidates = [result['candidate_'+key].copy() for key in ('boxes','scores','labels')]
                    if 'geometry_candidate_boxes' in result:
                        candidates[0] = result['geometry_candidate_boxes'].copy()
                if road_contract is not None:road_contract=road_memory.update(road_contract,info)
                reliability = _hugsim_expand(json.loads(Path(response['reliability']).read_text()))
                reliability = obstacle_memory.update(reliability, *candidates, info, current_occupancy=np.asarray(occupancy).reshape(5,200,200)[0])
                depth_points,depth_audit=depth_memory.update(obs,info)
                reliability['unknown_visibility_obstacles_xy']=depth_audit.pop('unknown_visibility_points_xy',[])
                reliability['unknown_visibility_geometry']=depth_audit.get('unknown_visibility_geometry',{})
                reliability['observed_depth_obstacles_xy']=depth_points.tolist()
                reliability['observed_depth_geometry']=depth_audit
                atomic_json(folder/'depth_geometry.json',depth_audit)
                reliability['require_predicted_drivable_area']=bool(dynamic)
                reliability['predicted_road_constraint_mode']='soft'
                anchored_reference,anchor_audit=navigation_anchor.update(plan,info)
                reliability['navigation_anchor_reference']=anchored_reference.tolist()
                reliability['navigation_anchor_audit']=anchor_audit
                # Follow the perceived detour as a spatial guide. The saved
                # path never exempts an obstacle: optimize() rechecks all
                # current occupancy and actor proposals at every frame.
                reliability['spatial_path_tracking']=True
                committed_reference,commitment_audit=committed_route.reference(plan,info,vehicle)
                if committed_reference is not None:
                    reliability['committed_path_center_xy']=committed_route.center_path(info).tolist()
                    reliability['route_commitment_audit']=commitment_audit
                if road_contract is not None:
                    reliability['predicted_drivable_area']=road_contract
                    atomic_json(folder/'predicted_drivable_audit.json',dict(DrivableArea(road_contract,vehicle).audit(),constraint_mode='soft',physical_road_boundary_verified=False))
                recovery_cap=controller.speed_cap(reliability)
                if recovery_cap is not None:
                    previous_cap=reliability.get('speed_limit_mps')
                    reliability=dict(reliability,speed_limit_mps=min(recovery_cap,previous_cap) if previous_cap is not None else recovery_cap)
                atomic_json(folder/'guard_memory_report.json', reliability)
                try: optimized = optimize(plan,occupancy,info,warm=warm,reliability=reliability,vehicle=vehicle,_reuse_verified_warm=bool(index%replan_ticks))
                except ValueError as exc:
                    if not str(exc).startswith('Empty occupancy:'): raise
                    optimized = dict(selected=None,reason=str(exc),ground_truth_used_by_optimizer=False)
                atomic_json(folder/'optimization.json',optimized)
                if optimized.get('selected') is None:committed_route.invalidate()
                else:committed_route.commit(optimized,info)
                action,decision=controller.update(optimized,motion.speed,timestamp,vehicle.dt,vehicle.emergency_braking_mps2)
                atomic_json(folder/'control_decision.json',decision)
                selected=optimized['selected']
                following=decision['state_after'] in ('normal','caution','recovering')
                warm=np.asarray(selected['controls']) if selected is not None and following else None
                mode=(decision['state_after']+' / MPC') if following else (decision['state_after']+' / reassessing')
                validate_action(dict(acc_mps2=action['acc'],steer_rate_radps=action['steer_rate']),vehicle,motion.speed,motion.steer)
                before = np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer])
                report['controls_attempted']=index+1
                road_clearance_after=None
                road_swept_after=None
                if road_contract is not None:
                    offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'],dtype=float)[:2,3]
                    local_initial=np.r_[-offset,0.,motion.speed,motion.steer]
                    local_after=integrate_state(local_initial,[action['acc'],action['steer_rate']],vehicle.dt,vehicle.wheelbase_m)
                    road_clearance_after=float(DrivableArea(road_contract,vehicle).clearance(local_after[None])[0])
                    road_swept_after=float(DrivableArea(road_contract,vehicle).swept_clearance(np.stack([local_initial,local_after])).min())
                    if reliability.get('predicted_road_constraint_mode','hard')=='hard' and road_swept_after < -1e-5:
                        report['preactuation_rejection']=dict(step=index,source_frame=source_frame,
                            action_not_executed=action,current_speed_mps=motion.speed,
                            initial_road_clearance_m=float(DrivableArea(road_contract,vehicle).clearance(local_initial[None])[0]),
                            endpoint_road_clearance_m=road_clearance_after,continuous_road_clearance_m=road_swept_after)
                        atomic_json(output/'preactuation_rejection.json',report['preactuation_rejection'])
                        raise RuntimeError('Next continuous footprint outside predicted road; diagnostic stopped before actuation')
                motion.advance(action)
                after = np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer])
                expected = integrate_state(before,[action['acc'],action['steer_rate']],vehicle.dt,vehicle.wheelbase_m)
                np.testing.assert_allclose(after,expected,rtol=0,atol=1e-10)
                for u in np.linspace(0,1,11):
                    p = before*(1-u)+after*u
                    check = reference_check(initial_native@native_delta(*p[:3]),reference,vehicle) if edited else dict(overlap=False,gap_m=None)
                    if edited:min_gap = min(min_gap,check['gap_m'])
                    if check['overlap']: raise RuntimeError('Interpolated reference overlap after control')
                    checked_native=initial_native@native_delta(*p[:3])
                    checked_source=T.T@checked_native@T
                    native_result=native_evaluation.check(checked_source,source_frame,vehicle)
                    native_gap=native_result['nearest']['gap_m']
                    if native_gap is not None:report['minimum_native_reference_gap_m']=min(report.get('minimum_native_reference_gap_m',float('inf')),native_gap)
                    actor_result=actor_evaluation.check(checked_source,min(source_frame+float(u),spec.frames-1),vehicle)
                    actor_gap=actor_result['nearest']['gap_m']
                    if actor_gap is not None:report['minimum_all_actor_reference_gap_m']=min(report.get('minimum_all_actor_reference_gap_m',float('inf')),actor_gap)
                    if actor_result['overlap']:
                        report['terminal_actor_reference']=dict(actor_result,source_frame=min(source_frame+float(u),spec.frames-1),ego_to_world=np.asarray(checked_source).tolist())
                        raise RuntimeError('Interpolated actor reference overlap after control')
                    if native_result['overlap']:
                        raise RuntimeError('Interpolated native-vehicle reference overlap after control')
                temporal = _hugsim_expand(json.loads((folder/'temporal_state.json').read_text()))
                if temporal['history_bev_present'] != (index > 0) or not temporal['metadata_cache_is_tracking_cache']:
                    raise RuntimeError('Unexpected temporal BEV state')
                report['trace'].append(dict(step=index,timestamp=timestamp,action=action,mode=mode,
                    optimization_reason=optimized.get('reason'),control_decision=decision,state_before=before.tolist(),state_after=after.tolist(),
                    neural_plan=plan.tolist(),perception=response['perception'],reliability=response['reliability'],
                    temporal=temporal,physics_integration_verified=True,predicted_road_constraint_mode='soft',predicted_road_footprint_clearance_after_m=road_clearance_after,predicted_road_swept_clearance_m=road_swept_after))
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
            if edited:report['minimum_inserted_reference_gap_m'] = min_gap
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),termination_reason='exception:'+type(exc).__name__)
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
        if extended and edited:
            from trajectory_baseline_score import score_report
            report['trajectory_score']=score_report(report,reference,vehicle)
            atomic_json(output/'trajectory_score.json',report['trajectory_score'])
        if extended:
            if report['status']=='failed':
                count=len(list((output/'comparison_frames').glob('*.png')))
                if count:
                    try:
                        report['video']=encode(output,spec.fps,count,filename=f'{scene}_{arm}_closed_three_views_partial.mp4')
                        report['video_partial']=True
                    except Exception as exc: report['video_error']=repr(exc)
        atomic_json(output/'run_report.json',report)
    if not report['implementation_unchanged']: raise RuntimeError('Control implementation changed during run')
    if not report['checkpoint_unchanged']: raise RuntimeError('Frozen checkpoint changed')
    executed=len(report['trace'])
    report['steps_executed']=executed
    report['video'] = encode(output,spec.fps,executed+1,filename=f'{scene}_{arm}_closed_three_views_{executed*vehicle.dt:g}s.mp4') if extended else encode(output,spec.fps,steps+1)
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
        report['checks']['no_background_freeze']=not any(row['background_frozen'] for row in report['observations'])
        if dynamic and not report['checks']['no_background_freeze']:raise RuntimeError('Dynamic validation froze source time')
        report['checks']['continuous_model_clock']=all(np.isclose(row['timestamp'],(start+i)*vehicle.dt) for i,row in enumerate(report['trace']))
        if any(count != executed+1 for count in report['checks']['three_camera_frame_counts'].values()) or not report['checks']['continuous_model_clock']:
            raise RuntimeError('Extended camera or model clock mismatch')
    if (not extended and not report['checks']['first_images_identical']) or not report['checks']['first_poses_equal']:
        raise RuntimeError('Comparison start mismatch')
    last = report['observations'][-1]
    report['final_open_closed_position_difference_m'] = float(np.linalg.norm(
        np.asarray(last['open_ego_to_world'])[:3,3]-np.asarray(last['closed_ego_to_world'])[:3,3]))
    report.update(status='completed',wall_seconds=time.monotonic()-began)
    if extended and edited:
        from trajectory_baseline_score import score_report
        report['trajectory_score']=score_report(report,reference,vehicle)
        atomic_json(output/'trajectory_score.json',report['trajectory_score'])
    atomic_json(output/'run_report.json',report)
    atomic_json(output/'progress.json',dict(stage='completed',video=report['video']['path']))
    return report
