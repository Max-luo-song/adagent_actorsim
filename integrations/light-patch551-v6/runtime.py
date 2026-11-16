"""Fresh observations -> frozen worker -> external reference -> light tracker -> native Motion."""

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

import copy,hashlib,importlib.util,json,os,pickle,subprocess,time,traceback
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from common import *
from tracker import Tracker
from environment import actor_arrays,Evaluation
from intervention import install_trajectory

TERMINATION=dict(simulation_observation_limit_s=18.8,drive_wall_limit_s=1800.,setup_wall_limit_s=2400.,drive_budget_starts='after worker ready',worker_boot_wall_limit_s=1200.,
    native_score_wall_limit_s=900.,
    per_inference_wall_limit_s=300.,collision='unchanged native bg >100 points and native polygon intersection; stop after detection',
    route_distance_limit_m=10.,route_completion_stop=1.,stop_speed_mps=.2,stop_hold_s=1.,
    stagnation_hold_s=3.,stagnation_displacement_m=.3,abnormal_stall_reference_speed_mps=2.,
    abnormal_stall_definition='stagnant >=3s while adopted reference asks >=2m/s during last >=1s; otherwise stopped/unresolved, not automatically a failure',
    recovery_speed_mps=1.,recovery_hold_s=1.,stall_does_not_terminate=True,
    normal_A_min_progress_ratio=.95,normal_A_max_extra_stopped_s=1.,normal_A_max_native_execution_hd_drop=.01,
    actual_hd_road_boundary_available=False,road_departure_reporting='native executed-path DAC proxy plus route distance; no invented HD boundary')

def dump_pickle(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,protocol=4)
    tmp.replace(path)

def wait_json(path,process,seconds):
    start=time.monotonic();last_notice=-15.;path=Path(path)
    while not path.exists():
        if process.poll() is not None:raise RuntimeError('Frozen UniAD worker exited: '+str(process.returncode))
        if time.monotonic()-start>seconds:raise TimeoutError('Worker protocol timeout: '+str(path))
        elapsed=time.monotonic()-start
        if elapsed-last_notice>=15:
            stage='waiting'
            startup=path.parent/'startup.json'
            if startup.exists():
                events=read(startup).get('events',[])
                if events:stage=events[-1]['stage']
            print('UniAD',path.name,'waiting',round(elapsed),'/',seconds,'s; stage=',stage,'pid=',process.pid,flush=True);last_notice=elapsed
        time.sleep(.1)
    result=read(path)
    if result.get('status')=='error':raise RuntimeError(str(result))
    return result

def start_worker(folder,out,seed):
    if not (folder/'step_000/observation.pkl').is_file():
        raise FileNotFoundError('Frozen UniAD bootstrap requires the actual first observation: '+str(folder/'step_000/observation.pkl'))
    folder.mkdir(parents=True,exist_ok=True);log=(out/'uniad_worker.log').open('x')
    env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',BEV_EXPERIMENT_SEED=str(seed),
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1',UNIAD_STARTUP_REPORT=str(folder/'startup.json'))
    for key in ['LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']:env.pop(key,None)
    from local_boot import local_archive
    support=local_archive(out.parent/'boot/manifest.json');env['PYTHONPATH']=support
    print('Launching resident UniAD with verified local Python support; log:',out/'uniad_worker.log',flush=True)
    process=subprocess.Popen([str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),str(HERE/'worker_trace.py'),
        '--job',str(folder),'--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:ready=wait_json(folder/'ready.json',process,TERMINATION['worker_boot_wall_limit_s'])
    except BaseException:
        process.terminate()
        try:process.wait(10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close();raise
    return process,log,ready

def stop_worker(folder,process,log):
    if process is not None:
        write(folder/'stop.json',dict(stop=True))
        try:process.wait(10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    if log is not None:log.close()

def processor(raw,perception,ego,state,candidate):
    if candidate is None:return np.asarray(raw).copy(),dict(source='raw_frozen_uniad',candidate_count=1,selected='raw_frozen_uniad',rejections=[],patch=False,reference_changed=False),state
    verify(candidate['module']);scope=importlib.util.spec_from_file_location('declared_external_patch',candidate['module']['path']);module=importlib.util.module_from_spec(scope);scope.loader.exec_module(module)
    allowed={k:np.asarray(v).copy() for k,v in perception.items() if k in candidate['allowed_perception_fields']}
    result=module.apply(plan=np.asarray(raw).copy(),perception=allowed,ego=copy.deepcopy(ego),state=copy.deepcopy(state))
    adopted=np.asarray(result['reference'],dtype=float)
    if adopted.shape!=(6,2) or not np.isfinite(adopted).all():raise ValueError('External patch returned invalid six-node reference')
    return adopted,dict(result.get('audit',{}),patch=True,module=candidate['module'],reference_changed=not np.allclose(adopted,raw,rtol=0,atol=1e-8)),result.get('state')

def prefix_gate(folder,common_folder,index,pose,images,perception,computed_action=None):
    previous=read(common_folder/f'steps/{index:03d}/receipt.json')
    if not np.allclose(pose,previous['ego_to_world'],rtol=0,atol=1e-9):raise ValueError('Common prefix actual pose mismatch')
    pixels={k:images[k]==previous['image_sha256'][k] for k in images}
    if not all(pixels.values()):raise ValueError('Common prefix camera pixels differ')
    with np.load(common_folder/f'uniad_worker/step_{index:03d}/perception.npz') as saved:
        if set(saved.files)!=set(perception):raise ValueError('Common prefix perception schema differs')
        deltas={}
        for k,v in perception.items():
            a=saved[k]
            if a.shape!=v.shape:raise ValueError('Common prefix perception shape differs: '+k)
            if a.dtype.kind in 'fc':
                delta=float(np.max(np.abs(a-v))) if a.size else 0.
                if not np.isfinite(delta) or delta>1e-4:raise ValueError('Common prefix neural output differs: '+k)
                deltas[k]=delta
            elif not np.array_equal(a,v):raise ValueError('Common prefix categorical output differs: '+k)
    # Verify the read-only motion-head export as well as the final perception packet.
    motion_deltas={}
    with np.load(common_folder/f'uniad_worker/step_{index:03d}/motion_head_raw.npz') as old, np.load(folder.parent.parent/f'uniad_worker/step_{index:03d}/motion_head_raw.npz') as new:
        if set(old.files)!=set(new.files):raise ValueError('Common prefix motion schema differs')
        for key in old.files:
            a,b=old[key],new[key]
            if a.shape!=b.shape:raise ValueError('Common prefix motion shape differs: '+key)
            if a.dtype.kind in 'fc':
                delta=float(np.max(np.abs(a-b))) if a.size else 0.
                if not np.isfinite(delta) or delta>1e-4:raise ValueError('Common prefix motion differs: '+key)
                motion_deltas[key]=delta
            elif not np.array_equal(a,b):raise ValueError('Common prefix motion categorical values differ: '+key)
    # Already-executed common history is replayed, never future references/poses.
    action=previous.get('action')
    if computed_action is not None and (action is None or not np.allclose([computed_action[k] for k in sorted(computed_action)],[action[k] for k in sorted(computed_action)],rtol=0,atol=1e-5)):
        raise ValueError('Common prefix light-controller action differs')
    receipt=dict(index=index,pixel_identical=pixels,neural_max_absolute_delta=deltas,motion_max_absolute_delta=motion_deltas,
        pose_tolerance=1e-9,neural_tolerance=1e-4,action_tolerance=1e-5,
        full_model_history_rebuilt_by_fresh_prefix_inference=True,partial_tensor_snapshot_used=False,
        executed_common_past_action_reused=action is not None)
    write(folder/'prefix_gate.json',receipt);return action

def run_arm(root,name,steps,common=None,candidate=None):
    root=Path(root);out=root/name
    if name not in ('B_repeat','B_patch_on','A_patch_on') or common is None or (name.endswith('patch_on') != (candidate is not None)):raise ValueError('Use declared repeat or patch validation arms only')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);configure()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no closed-loop simulation executed')
    from native_car_edit import make_types
    from live_env import Motion
    from uniad_bridge import to_uniad,ORDER
    from build_source_sequence import training_lidar_mount
    from scene_api import SceneSpec
    manifest=root/'scene_manifest.json';spec=SceneSpec.read(manifest);scenario=read(root/'scenario/scenario.json')['config']
    baseline=read(root/'baseline_freeze.json');verify(baseline['verification'])
    from map_reference import MapRoadReference
    road_reference=MapRoadReference(spec.vehicle,read(root/'map/map.json'))
    plan=read(root/'plan.json');prefix_end=int(round(scenario['intervention_start_s']*spec.fps))
    Backend,Session=make_types();session=None;process=log=None;worker=out/'uniad_worker';began=time.monotonic()
    report=dict(schema='fixed_light_tracking_closed_loop_v1',arm=name,status='initializing',trace=[],observations=[],timing={},
        seed=plan['seed'],tracker=baseline,steps_budget=steps,termination=TERMINATION,
        scene=551,patch=candidate,real_closed_loop=True,initial_context=plan['initial_context'],source_clock='initial source offset + local simulation time; model timestamps use local clock',heavy_mpc_executed=False,
        default_driver_changed=False,uniad_history_strategy='full fresh common-prefix replay; never reset at intervention',
        truth_used_online=True,truth_scope='user-authorized static map only',actor_truth_used_by_controller=False,recorded_future_ego_used_by_controller=False,route_read_by_controller=False,nav_command=2,nav_command_source='fixed straight navigation command',
        controls_selected_by='same static map -> declared external reference retiming -> unchanged light iLQR -> same map gate')
    report['vehicle_geometry']=dict(half_length_m=spec.vehicle.half_length_m,half_width_m=spec.vehicle.half_width_m,
        body_height_m=spec.vehicle.body_height_m,clearance_margin_m=spec.vehicle.clearance_margin_m)
    motion=Motion(spec.vehicle.dt,spec.vehicle.wheelbase_m,scenario['initial_ego_speed_mps']);tracker=Tracker(spec.vehicle)
    mount=training_lidar_mount(ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml');anchor=np.asarray(plan['initial_context']['ego_to_world'],dtype=float);source_frame_offset=int(plan['initial_context']['source_frame']);processing_state=None
    drive_began=None
    write(out/'run_report.json',report);write(out/'progress.json',dict(stage='loading_scene'))
    try:
        report['current_stage']='scene_load';load_began=time.monotonic();session=Session(backend_factory=Backend);session.load_scene(manifest);session.vehicles.activation_frame=prefix_end
        report['timing']['scene_load_seconds']=time.monotonic()-load_began
        if time.monotonic()-began>TERMINATION['setup_wall_limit_s']:raise TimeoutError('Scene setup wall limit exceeded before driving')
        if name.startswith('B'):
            report['actor_intervention']=install_trajectory(session,root/'scenario/target_track.npz',scenario['absolute_source_activation_s'])
        else:report['actor_intervention']=dict(applied=False,original_native_behavior=True)
        tracks=actor_arrays(session);np.savez_compressed(out/'actor_tracks.npz',**tracks);report['actor_tracks_asset']=asset(out/'actor_tracks.npz')
        from validation import actor_sync
        report['actor_sync']=actor_sync(root,name,tracks,session)
        background_opacity=session.backend.trainer.models['Background'].get_opacity.detach().cpu().numpy()
        evaluation=Evaluation(tracks,spec.vehicle,background_opacity);report['rigid_actor_count']=len(tracks['RigidNodes__instances_size'])
        if report['rigid_actor_count']!=15:raise ValueError('Native actor count changed')
        for index in range(steps+1):
            report['last_step_index']=index;report['current_stage']='current_observation_render'
            if drive_began is not None and time.monotonic()-drive_began>TERMINATION['drive_wall_limit_s']:
                report.update(status='resource_limited',termination_reason='wall_clock_limit');break
            timestamp=index*spec.vehicle.dt;source_frame=index+source_frame_offset;
            if source_frame>plan['source_end_frame']:raise ValueError('Source frame outside valid episode; never clamp future actors')
            source_timestamp=source_frame/spec.fps;pose=tracker.helpers['motion_pose'](anchor,motion)
            write(out/'progress.json',dict(stage='live_light_tracking',index=index,total_controls=steps,simulation_time_s=timestamp))
            render_began=time.monotonic();packet=session.render(pose,source_timestamp);render_seconds=time.monotonic()-render_began
            folder=out/f'steps/{index:03d}';folder.mkdir(parents=True)
            image_hash={}
            for camera in ORDER:
                path=folder/(camera+'.png');Image.fromarray(packet['cameras'][camera]['rgb']).save(path);image_hash[camera]=sha(path)
            # Equal trust-only context in both arms; real intervention is recorded separately.
            model_packet=dict(packet,timestamp_s=timestamp,edit_trust=dict(edited=True,status='diagnostic_unverified',formal_evaluation_admitted=False))
            obs,info=to_uniad(model_packet,mount,speed_mps=motion.speed,steering_rad=motion.steer,command=2,
                scene_token=plan['common_scene_token'],dt=spec.vehicle.dt,
                acceleration_mps2=motion.last_accel,steering_rate_radps=motion.last_rate)
            info.update(closed_loop_run=True,control_actions_executed=True,ego_kinematics_source='native_motion_control_feedback')
            state=np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer]);check=evaluation.check(pose,source_frame)
            map_gap=road_reference.geometry.pose_clearance(pose)
            check.update(authorized_static_map_clearance_m=map_gap,authorized_static_map_departure=map_gap<0)
            receipt=dict(index=index,timestamp_s=timestamp,source_frame=source_frame,source_timestamp_s=source_timestamp,ego_to_world=pose.tolist(),motion_state=state.tolist(),image_sha256=image_hash,render_seconds=render_seconds,
                environment=check,action=None)
            report['observations'].append(dict(index=index,timestamp=timestamp,pose=pose.tolist(),motion_state=state.tolist(),environment=check,image_sha256=image_hash))
            if check['authorized_static_map_departure'] or check['collision'] or check['native_route_deviation'] or check['native_route_completed']:
                reason='authorized_static_map_departure' if check['authorized_static_map_departure'] else 'collision' if check['collision'] else 'native_route_deviation' if check['native_route_deviation'] else 'native_route_complete'
                report.update(status='terminated',termination_reason=reason);write(folder/'receipt.json',receipt);break
            stepdir=worker/f'step_{index:03d}';dump_pickle(stepdir/'observation.pkl',(obs,info));dump_pickle(stepdir/'sensor_metadata.pkl',info)
            if process is None:
                report['current_stage']='frozen_worker_boot';boot_began=time.monotonic()
                write(out/'progress.json',dict(stage='booting_resident_UniAD',log=str(out/'uniad_worker.log')))
                process,log,ready=start_worker(worker,out,plan['seed']);report['worker_ready']=ready
                report['timing']['worker_boot_seconds']=time.monotonic()-boot_began
                if time.monotonic()-began>TERMINATION['setup_wall_limit_s']:raise TimeoutError('Setup wall limit exceeded before driving')
                drive_began=time.monotonic();report['timing']['setup_seconds']=drive_began-began
                if ready['planning_bev']!='current' or ready.get('branch_restored') is not None:raise ValueError('Unexpected neural worker/history startup')
                if ready.get('model_profile')!='calibration_geometry_v1' or ready.get('original_collision_postprocessor') is not False or ready.get('cameras')!=3 or ready.get('thresholds')!=[.4,.35]:
                    raise ValueError('Frozen UniAD input/head profile changed')
            request=dict(index=index,packet=str(stepdir/'observation.pkl'))
            if index==0:request['diagnostic_seed']=plan['seed']
            write(worker/f'request_{index:03d}.json',request)
            report['current_stage']='frozen_uniad_inference';response=wait_json(worker/f'done_{index:03d}.json',process,TERMINATION['per_inference_wall_limit_s'])
            if not np.isclose(response['timestamp'],timestamp):raise ValueError('Frozen worker source clock mismatch')
            with np.load(response['perception']) as z:perception={k:z[k].copy() for k in z.files}
            temporal=read(stepdir/'temporal_state.json')
            if temporal['history_bev_present']!=(index>0) or temporal['metadata_cache_is_tracking_cache'] is not True:raise ValueError('UniAD history was reset or incomplete')
            raw=np.asarray(perception['trajectory']);active_candidate=candidate
            patch_perception=dict(perception)
            if not (stepdir/'motion_head_raw.npz').exists():raise RuntimeError('Read-only motion prediction export missing')
            with np.load(stepdir/'motion_head_raw.npz') as z:
                for key in z.files:patch_perception['motion_'+key]=z[key].copy()
            patch_perception['motion_sdc_last_query']=np.asarray(True)
            report['current_stage']='external_reference_processing'
            mapped,selection=road_reference.process(raw,perception,obs,info,motion)
            from replay import ego_inputs
            adopted,patch_audit,processing_state=processor(mapped,perception,ego_inputs(info,motion,spec.vehicle),processing_state,candidate)
            if candidate is not None:
                patch_certificate=road_reference.certificate(road_reference.states(adopted))
                road_reference.reference_valid=patch_certificate['passed']
                selection['reference_certificate_after_patch']=patch_certificate
            selection['external_patch']=patch_audit
            selection['reference_changed']=not np.allclose(adopted,raw,rtol=0,atol=1e-8)
            write(folder/'patch_receipt.json',dict(audit=patch_audit,before_patch_reference=mapped.tolist(),after_patch_reference=adopted.tolist(),
                information_scope='retained neural detections and current actual ego state; no target script or future actor truth'))
            write(folder/'reference.json',dict(raw_uniad=raw.tolist(),before_patch_reference=mapped.tolist(),adopted=adopted.tolist(),selection=selection,
                runtime_inputs='same static map, current UniAD plan/detections, bounded past detection memory, actual ego state; no actor truth or recorded future ego trajectory'))
            if not (stepdir/'motion_head_raw.npz').exists():raise RuntimeError('Read-only motion prediction export missing')
            receipt.update(raw_uniad=raw.tolist(),adopted=adopted.tolist(),patch_applied=bool(patch_audit['reference_changed']),patch_audit=patch_audit,temporal=temporal,inference_seconds=response.get('inference_seconds'),
                external_reference_changed=selection['reference_changed'],
                perception_asset=asset(stepdir/'perception.npz'),motion_prediction_asset=asset(stepdir/'motion_head_raw.npz'),
                reference_asset=asset(folder/'reference.json'))
            if index==steps:
                if common and index<=prefix_end:prefix_gate(folder,common,index,pose,image_hash,perception)
                write(folder/'receipt.json',receipt);report.update(status='completed',termination_reason='source_scene_time_window_complete');break
            report['current_stage']='fixed_light_tracking';action,control=tracker.control(adopted,info,motion)
            nominal_action=action.copy();action,road_gate=road_reference.admit(motion,nominal_action)
            receipt['road_gate']=road_gate
            if action is None:
                receipt['road_reference']=selection;write(folder/'receipt.json',receipt)
                report.update(status='terminated',termination_reason='no_verified_road_safe_action',road_safety_termination=road_gate);break
            receipt['road_reference']=selection
            if common and index<=prefix_end:
                historical=prefix_gate(folder,common,index,pose,image_hash,perception,action)
                if historical is None:raise ValueError('Common prefix lacks an executed historical action')
                action=historical
            report['current_stage']='native_motion_and_environment_checks';before=state.copy();motion.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']))
            after=np.array([motion.x,motion.z,motion.yaw,motion.speed,motion.steer]);swept=[]
            delta=np.array([after[1]-before[1],after[0]-before[0]])
            c,s=np.cos(before[2]),np.sin(before[2]);body_after=delta@np.array([[c,-s],[s,c]])
            desired=np.asarray(control['reference_states'])[1]
            receipt['tracking_error']=dict(position_m=float(np.linalg.norm(body_after-desired[:2])),
                lateral_right_m=float(body_after[1]-desired[1]),longitudinal_forward_m=float(body_after[0]-desired[0]),
                heading_rad=float(after[2]-before[2]-desired[2]),evaluation_time_s=.1)
            for fraction in np.linspace(.1,1.,10):
                interpolated=before+(after-before)*fraction
                proxy=copy.copy(motion);proxy.x,proxy.z,proxy.yaw=interpolated[:3]
                swept_pose=tracker.helpers['motion_pose'](anchor,proxy);result=evaluation.check(swept_pose,source_frame+fraction)
                result.update(timestamp_s=(index+fraction)*spec.vehicle.dt,source_frame=source_frame+fraction,interpolation_fraction=float(fraction),ego_to_world=swept_pose.tolist())
                swept.append(result)
                if result['collision']:break
            receipt.update(action=action,control=control,state_after=after.tolist(),swept_collision=any(r['collision'] for r in swept))
            write(folder/'receipt.json',receipt);report['trace'].append(receipt)
            if index%10==0:write(out/'run_report.json',report)
            print(name,index+1,'/',steps,'speed',round(motion.speed,3),flush=True)
            if receipt['swept_collision']:
                # Match the environment's post-action observation even on a collision stop.
                end_pose=tracker.helpers['motion_pose'](anchor,motion);end_packet=session.render(end_pose,(source_frame+1)/spec.fps)
                end_folder=out/f'steps/{index+1:03d}';end_folder.mkdir(parents=True,exist_ok=True);end_hash={}
                for camera in ORDER:
                    path=end_folder/(camera+'.png');Image.fromarray(end_packet['cameras'][camera]['rgb']).save(path);end_hash[camera]=sha(path)
                end_check=evaluation.check(end_pose,source_frame+1)
                terminal=dict(index=index+1,timestamp=(index+1)*spec.vehicle.dt,pose=end_pose.tolist(),motion_state=after.tolist(),environment=end_check,image_sha256=end_hash)
                report['observations'].append(terminal)
                write(end_folder/'receipt.json',dict(index=index+1,timestamp_s=(index+1)*spec.vehicle.dt,ego_to_world=end_pose.tolist(),motion_state=after.tolist(),image_sha256=end_hash,environment=end_check,action=None,terminal_observation=True))
                report.update(status='terminated',termination_reason='swept_environment_collision',terminal_collision=swept[-1]);break
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),termination_reason='runtime_error');(out/'error.log').write_text(traceback.format_exc())
    finally:
        stop_worker(worker,process,log)
        if session is not None:
            try:session.close()
            except BaseException as exc:report['scene_cleanup_error']=repr(exc)
        session=None
        import gc
        gc.collect();torch.cuda.empty_cache()
        report.update(driving_wall_seconds=(time.monotonic()-drive_began if drive_began is not None else 0.),wall_seconds=time.monotonic()-began,controls_completed=len(report['trace']),model_weights_unchanged=None,
            model_integrity_note='outer stage verifies pinned code and weight hashes after this arm')
        write(out/'run_report.json',report)
    return report
