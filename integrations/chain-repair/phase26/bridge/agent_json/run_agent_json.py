"""Batch JSON-agent emulator or persistent JSONL request/response process."""

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

import argparse
from contextlib import redirect_stdout
import hashlib
import json
pass  # Path is provided by the portable boundary above
import sys
import agent_bootstrap as bootstrap
from agent_bridge import AgentBridge
from agent_catalog import Catalog,strict_json
from agent_jobs import validate_job,run_job,references


def serve_jsonl(bridge,stream,output):
    """Protocol responses only on output; Python model logs go to stderr."""
    for line in stream:
        if not line.strip():continue
        try:
            value=strict_json(line)
            with redirect_stdout(sys.stderr):response=bridge.handle(value)
        except (ValueError,TypeError) as exc:
            response=dict(schema='agent_response_v1',request_id=None,status='error',error=dict(code='invalid_json',message=str(exc)),
                replayed=False,context=bridge.context(),training_performed=False,control_executed=False,formal_evaluation_admitted=False)
        output.write(json.dumps(response,allow_nan=False)+'\n');output.flush()
        if bridge.closed:break



def trajectory_preflight(job,catalog):
    """Compile literal trajectories against scene files before GPU construction."""
    from ego_trajectory import compile_trajectory,ffmpeg_path
    import numpy as np
    from scene_edit_session import saf
    scene=None;results=[]
    for step in job['steps'] if job else []:
        req=step['request'];args=req['args']
        if req['action']=='load_scene':
            scene=args.get('scene_id') if isinstance(args.get('scene_id'),str) else None
        if req['action'] not in ('validate_ego_trajectory','preview_ego_trajectory','validate_ego_execution','execute_ego_trajectory'):continue
        if step.get('expect',{}).get('status')=='error':continue
        if scene is None or list(references(args.get('trajectory',{}))):
            results.append(dict(request_id=req['request_id'],compiled=False,reason='requires runtime response reference'));continue
        spec=saf.SceneSpec.read(catalog.scene(scene))
        compiled=compile_trajectory(args.get('trajectory'),spec)
        if req['action'] in ('preview_ego_trajectory','execute_ego_trajectory'):ffmpeg_path()
        if req['action'] in ('validate_ego_execution','execute_ego_trajectory'):
            from ego_execution import execution_plan,make_solver
            execution_plan(compiled,spec.vehicle,args);make_solver(spec.vehicle)
        final=compiled['samples'][-1]; delta=np.linalg.inv(np.asarray(final['recorded_ego_to_world']))@np.asarray(final['ego_to_world'])
        results.append(dict(request_id=req['request_id'],compiled=True,scene_id=scene,
            final_distance_from_recorded_m=float(np.linalg.norm(delta[:3,3])),
            final_yaw_from_recorded_degrees=float(np.degrees(np.arctan2(delta[1,0],delta[0,0]))),
            sample_count=compiled['sample_count'],duration_s=compiled['duration_s'],fps=compiled['fps'],
            maximum_sampled_speed_mps=compiled['maximum_sampled_speed_mps']))
    return results



def signal_preflight(job,catalog):
    from traffic_lights import SIGNAL_OPERATIONS,emitter_candidates,validate_signal
    from scene_edit_session import saf
    import torch
    scene=None;cache={};results=[]
    for step in job['steps'] if job else []:
        req=step['request'];args=req['args']
        if req['action']=='load_scene':scene=args.get('scene_id') if isinstance(args.get('scene_id'),str) else None
        command=args.get('command',{})
        if req['action'] not in ('edit','validate_edit') or not isinstance(command,dict) or command.get('operation') not in SIGNAL_OPERATIONS:continue
        if step.get('expect',{}).get('status')=='error':continue
        if scene is None or list(references(command)):
            results.append(dict(request_id=req['request_id'],compiled=False,reason='requires runtime reference'));continue
        spec=saf.SceneSpec.read(catalog.scene(scene));path=spec.source['checkpoint']
        if command.get('object_id','').startswith('signal:'):
            from calibrated_signals import load_profiles,build_geometry
            from edit_api import read_command
            compiled=read_command(dict(command,schema='scene_edit_v1',mode='diagnostic_preview'))
            validate_signal(compiled,duration=spec.frames/spec.fps)
            key=command['object_id'].split(':',1)[1];profiles=load_profiles(spec)
            if key not in profiles:raise ValueError('calibrated signal profile unavailable')
            results.append(dict(request_id=req['request_id'],compiled=True,profile_id=key,gaussian_points=len(build_geometry(profiles[key])['means']),visual_gpu_verified=False))
            continue
        if path not in cache:
            data=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            model=data['models'].get('TrafficLightNodes')
            if model is None:raise ValueError('checkpoint has no native traffic lights')
            cache[path]=(model['points_ids'].flatten(),emitter_candidates(model['_features_dc']),model['instances_fv'].shape[1])
        ids,emit,count=cache[path]
        from edit_api import read_command
        compiled=read_command(dict(command,schema='scene_edit_v1',mode='diagnostic_preview'))
        validate_signal(compiled,duration=spec.frames/spec.fps)
        index=int(command['object_id'].split(':')[-1]);selected=int(((ids==index)&emit).sum())
        if not 0<=index<count or selected==0:raise ValueError('traffic signal target has no supported emitter candidate points')
        results.append(dict(request_id=req['request_id'],compiled=True,scene_id=scene,model_index=index,
            emitter_candidate_points=selected,visual_selection_verified=False,semantic_control_connected=False))
    return results


def actor_preflight(job,catalog):
    """Validate real checkpoint body availability without constructing a GPU model."""
    from actor_trajectories import compile_path
    from scene_edit_session import saf
    from edit_api import read_command
    import torch,yaml
    scene=None;cache={};results=[]
    for step in job['steps'] if job else []:
        req=step['request'];args=req['args']
        if req['action']=='load_scene':scene=args.get('scene_id') if isinstance(args.get('scene_id'),str) else None
        command=args.get('command',{})
        if req['action'] not in ('edit','validate_edit') or not isinstance(command,dict) or command.get('operation')!='set_actor_trajectory':continue
        if step.get('expect',{}).get('status')=='error':continue
        if scene is None or list(references(command)):
            results.append(dict(request_id=req['request_id'],compiled=False,reason='requires runtime reference'));continue
        spec=saf.SceneSpec.read(catalog.scene(scene));path=spec.source['checkpoint']
        read_command(dict(command,schema='scene_edit_v1',mode='diagnostic_preview'))
        if path not in cache:cache[path]=torch.load(path,map_location='cpu',mmap=True,weights_only=False)['models']['SMPLNodes']
        model=cache[path];index=int(command['object_id'].split(':')[-1])
        count=model['instances_trans'].shape[1]
        if not 0<=index<count:raise ValueError('actor target outside native body indices')
        points=int((model['points_ids'].flatten()==index).sum())
        if not points or 'smpl_qauts' not in model:raise ValueError('actor body has no source Gaussian/joint animation')
        # Runtime manager obtains test frames directly from the loaded trainer.
        # Read the identical source config rather than assuming a holdout cadence.
        config=spec.source.get('source_config')
        if config is None:raise ValueError('source configuration required for actor preflight')
        cfg=yaml.safe_load(Path(config).read_text());stride=cfg['data']['pixel_source'].get('test_image_stride',0)
        tests=range(stride,spec.frames,stride) if stride>0 else ()
        plan=compile_path(command['trajectory'],spec,model['instances_trans'][:,index].numpy(),model['instances_fv'][:,index].numpy().astype(bool),tests)
        motion=None
        if plan['motion_profile']=='smooth_v1':
            from actor_motion import prepare_motion
            motion=prepare_motion(plan,spec,model['instances_trans'][:,index].numpy(),model['instances_fv'][:,index].numpy().astype(bool),model['instances_quats'][:,index].numpy(),model['smpl_qauts'][:,index].numpy(),tests)
        results.append(dict(request_id=req['request_id'],compiled=True,model_index=index,gaussian_points=points,
            motion_profile=plan['motion_profile'],gait_retimed=motion is not None,
            joint_roughness_before=motion['joint_roughness_before'] if motion else None,joint_roughness_after=motion['joint_roughness_after'] if motion else None,velocity_jump_after_mps=motion['velocity_jump_after_mps'] if motion else None,velocity_jump_before_mps=motion['velocity_jump_before_mps'] if motion else None,playback_rate_range=[motion['min_playback_rate'],motion['max_playback_rate']] if motion else None,last_animation_frame=motion['last_animation_frame'] if motion else None,
            first_frame=plan['first_frame'],last_frame=plan['last_frame'],max_speed_mps=plan['max_speed_mps'],
            body_animation_present=True,gpu_render_verified=False))
    return results


def actor_preview_preflight(job,catalog):
    from actor_motion_review import load_baseline
    from scene_edit_session import saf
    scene=None;results=[]
    for step in job['steps'] if job else []:
        req=step['request'];args=req['args']
        if req['action']=='load_scene':scene=args.get('scene_id') if isinstance(args.get('scene_id'),str) else None
        if req['action']!='preview_actor_trajectory' or step.get('expect',{}).get('status')=='error':continue
        if scene is None or any(list(references(args.get(k))) for k in ('object_id','start_frame','end_frame','camera_source_frame','comparison_baseline')):continue
        if 'comparison_baseline' in args:
            spec=saf.SceneSpec.read(catalog.scene(scene))
            baseline=load_baseline(args['comparison_baseline'],spec,args['object_id'],args['start_frame'],args['end_frame'],args['camera_source_frame'])
            results.append(dict(request_id=req['request_id'],baseline_manifest=baseline['path'],available=True))
    return results


def main():
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--job',type=Path);modes.add_argument('--task',type=Path);modes.add_argument('--jsonl',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--catalog',type=Path)
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    catalog=Catalog(args.catalog)
    input_file=args.task or args.job
    input_value=strict_json(input_file.read_text()) if input_file else None
    task=input_value if input_value and input_value.get('schema')=='scene_task_v1' else None
    if args.task and task is None:raise ValueError('--task requires scene_task_v1')
    if task is not None:
        from scene_tasks import compile_task
        job=compile_task(task)
    else:job=validate_job(input_value) if input_value else None
    preflight=catalog.preflight()
    trajectory_checks=trajectory_preflight(job,catalog)
    signal_checks=signal_preflight(job,catalog)
    actor_checks=actor_preflight(job,catalog)
    actor_baselines=actor_preview_preflight(job,catalog)
    from task_preflight import weather_preflight
    weather_checks=weather_preflight(job)
    if task is not None:
        spec=__import__('scene_edit_session').saf.SceneSpec.read(catalog.scene(task['scene_id']))
        if any(frame>=spec.frames for frame in task.get('verification_frames',[])):raise ValueError('task verification frame outside scene')
    if args.check_only:
        print(json.dumps(dict(valid=True,input_schema=input_value.get('schema') if input_value else None,weather_preflight=weather_checks,steps=len(job['steps']) if job else None,catalog=preflight,
            scene_loaded=False,backend_target_existence_checked=False,execution_performed=False,
            actor_comparison_baselines=actor_baselines,actor_trajectory_preflight=actor_checks,traffic_light_preflight=signal_checks,trajectory_preflight=trajectory_checks,verification_requirements=job.get('verification',{}) if job else {}),indent=2));return
    if args.output is None:p.error('--output required for execution')
    if args.output.exists():raise FileExistsError('refusing to overwrite '+str(args.output))
    args.output.mkdir(parents=True,exist_ok=False);out=args.output.resolve()
    (out/'weather_preflight.json').write_text(json.dumps(weather_checks,indent=2))
    (out/'actor_comparison_baselines.json').write_text(json.dumps(actor_baselines,indent=2))
    (out/'actor_trajectory_preflight.json').write_text(json.dumps(actor_checks,indent=2))
    (out/'traffic_light_preflight.json').write_text(json.dumps(signal_checks,indent=2))
    (out/'trajectory_preflight.json').write_text(json.dumps(trajectory_checks,indent=2))
    (out/'catalog_preflight.json').write_text(json.dumps(preflight,indent=2))
    paths=list(bootstrap.HERE.glob('*.py'))+[catalog.path,bootstrap.BRIDGE/'unified/edit_api.py',
        bootstrap.BRIDGE/'unified/scene_edit_session.py',bootstrap.BRIDGE/'framework_backend/backend.py',
        bootstrap.BRIDGE/'framework_backend/native_edit.py',bootstrap.BRIDGE/'framework_backend/native_radiance.py',
        bootstrap.BRIDGE/'framework_backend/managed_assets.py',bootstrap.BRIDGE/'framework_backend/traffic_lights.py',bootstrap.BRIDGE/'framework_backend/calibrated_signals.py']
    paths += [bootstrap.BRIDGE/'framework_backend/actor_trajectories.py',bootstrap.BRIDGE/'framework_backend/actor_motion.py']
    paths += [bootstrap.ROOT/'waymo23-native-import/live_env.py',bootstrap.ROOT/'code/HUGSIM/sim/ilqr/lqr_solver.py',
        bootstrap.ROOT/'code/HUGSIM/sim/ilqr/utils.py',bootstrap.ROOT/'waymo23-native-import/configs/vehicle.json']
    paths+=list((bootstrap.BRIDGE/'framework_backend/signal_profiles').glob('*.json'))
    if input_file:paths.append(input_file.resolve());(out/'job.json').write_text(json.dumps(job,indent=2))
    if task is not None:(out/'task.json').write_text(json.dumps(task,indent=2))
    if weather_checks:paths += [bootstrap.BRIDGE/'weather_runtime'/name for name in ('runtime_backend.py','mask_backend.py','runtime_contract.py','weather_worker.py')]
    (out/'implementation.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},indent=2))
    bridge=AgentBridge(out/'session',catalog=catalog)
    try:
        if task is not None:
            from scene_tasks import execute_task
            result=execute_task(task,bridge,out/'task')
            (out/'task_result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2));print('Scene task output:',out,flush=True)
            if result['status']!='completed':raise RuntimeError('scene task failed; inspect task_result.json')
        elif job:
            result=run_job(job,bridge,out/'job_results')
            print(json.dumps(result,indent=2));print('Agent JSON output:',out,flush=True)
        else:
            serve_jsonl(bridge,sys.stdin,sys.stdout)
    finally:
        bridge.close()

if __name__=='__main__':main()
