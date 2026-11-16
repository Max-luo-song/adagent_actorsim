"""One declarative scene task compiled to the existing guarded request API.

Batch JSON and a future model use identical execution, receipts and artifacts.
Tasks are serial workflows, not atomic multi-edit transactions.
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

import copy,hashlib,json
pass  # Path is provided by the portable boundary above
from agent_bridge import BridgeError,fields,ID

from task_contracts import TASK_SCHEMA,RESULT_SCHEMA,PREVIEWS


def validate_task(value):
    from agent_jobs import references
    fields(value,{'schema','task_id','scene_id'},{'description','weather','edits','previews','ego_execution','verification_frames','reset_after','attitude_policy'})
    try:value=_hugsim_expand(json.loads(json.dumps(value,allow_nan=False)))
    except (ValueError,TypeError) as exc:raise ValueError('task must be finite JSON') from exc
    if value['schema']!='scene_task_v1':raise ValueError('expected scene_task_v1')
    if not isinstance(value['task_id'],str) or not ID.fullmatch(value['task_id']):raise ValueError('invalid task_id')
    if not isinstance(value['scene_id'],str) or not value['scene_id']:raise ValueError('scene_id must be a nonempty registered name')
    if 'description' in value and not isinstance(value['description'],str):raise ValueError('description must be a string')
    if value.get('weather','clear') not in ('clear','rainy'):raise ValueError('weather requires clear/rainy')
    if value.get('attitude_policy','gravity_level') not in ('gravity_level','initial_ego','recorded_up'):raise ValueError('invalid task attitude_policy')
    policy=value.get('attitude_policy','gravity_level')
    if type(value.get('reset_after',True)) is not bool:raise ValueError('reset_after must be boolean')
    if list(references(value)):raise ValueError('scene tasks require literal values; versions/references are assigned by executor')
    edits=value.get('edits',[]);previews=value.get('previews',[]);frames=value.get('verification_frames',[])
    if not isinstance(edits,list) or len(edits)>32:raise ValueError('edits requires a list of at most 32 commands')
    from edit_api import read_command
    for command in edits:
        if not isinstance(command,dict):raise ValueError('edit command must be an object')
        if set(command)&{'schema','mode','expected_generation','expected_revision'}:raise ValueError('task executor owns edit protocol metadata')
        operation=command.get('operation')
        if operation=='insert':
            fields(command,{'operation','object_id','asset_id'},{'placement_id','asset_to_world'})
            if not isinstance(command['object_id'],str) or not command['object_id'].startswith('managed:'):raise ValueError('insert requires managed object_id')
            if not isinstance(command['asset_id'],str):raise ValueError('asset_id must be a registered name')
            if ('placement_id' in command)==('asset_to_world' in command):raise ValueError('insert needs one placement')
        elif operation=='set_pose':
            fields(command,{'operation','object_id'},{'placement_id','asset_to_world'})
            if ('placement_id' in command)==('asset_to_world' in command):raise ValueError('set_pose needs one placement')
        else:read_command(dict(command,schema='scene_edit_v1',mode='diagnostic_preview'))
    if not isinstance(previews,list) or len(previews)>8:raise ValueError('previews requires at most 8 entries')
    for preview in previews:
        fields(preview,{'kind','args'})
        if preview['kind'] not in PREVIEWS or not isinstance(preview['args'],dict):raise ValueError('invalid preview kind/args')
        args=preview['args']
        if set(args)&{'expected_generation','expected_revision'}:raise ValueError('task executor owns preview versions')
        if preview['kind']=='camera':
            fields(args,set(),{'source_frame','timestamp_s','ego_to_world','attitude_policy'})
            if set(args)-{'attitude_policy'} not in ({'source_frame'},{'timestamp_s','ego_to_world'}):raise ValueError('camera preview needs source_frame OR timestamp_s and ego_to_world')
        elif preview['kind']=='actor_trajectory':fields(args,{'object_id','start_frame','end_frame','camera_source_frame'},{'render_fps','comparison_baseline','attitude_policy'})
        else:fields(args,{'trajectory'},{'compare_recorded','level_recorded_baseline'})
        if 'attitude_policy' in args and args['attitude_policy']!=policy:raise ValueError('preview attitude_policy conflicts with task policy')
        if 'trajectory' in args and args['trajectory'].get('attitude_policy',policy)!=policy:raise ValueError('trajectory attitude_policy conflicts with task policy')
        if 'level_recorded_baseline' in args and args['level_recorded_baseline']!=(policy=='gravity_level'):raise ValueError('recorded baseline attitude conflicts with task policy')
    if not isinstance(frames,list) or any(type(f) is not int or f<0 for f in frames) or len(frames)>8 or len(set(frames))!=len(frames):raise ValueError('verification_frames requires distinct nonnegative frame indices')
    if frames and not value.get('reset_after',True):raise ValueError('verification_frames requires reset_after=true')
    if 'ego_execution' in value:
        ego=value['ego_execution']
        fields(ego,{'driver','trajectory'},{'initial_speed_mps','initial_steer_rad','max_tracking_error_m','max_tracking_heading_error_degrees'})
        if ego['trajectory'].get('attitude_policy',policy)!=policy:raise ValueError('execution attitude_policy conflicts with task policy')
        if ego['driver']!='reference_tracking':raise ValueError('driver adapter unavailable; supported: reference_tracking; resident_uniad remains standalone')
    return copy.deepcopy(value)


def compile_task(value,*,context=None,namespace=None):
    from agent_jobs import validate_job
    task=validate_task(value);context=context or {};namespace=namespace or task['task_id']
    prefix='task_'+hashlib.sha256((namespace+'\0'+json.dumps(task,sort_keys=True)).encode()).hexdigest()[:12]
    steps=[];names={};current_scene=context.get('scene_id');weather=task.get('weather','clear');policy=task.get('attitude_policy','gravity_level')
    def add(name,action,args):
        identity=prefix+'_'+name;names[name]=identity
        steps.append(dict(request=dict(schema='agent_request_v1',request_id=identity,action=action,args=args)))
        return identity
    initial={k:context[k] for k in ('expected_generation','expected_revision') if k in context}
    if current_scene is not None:
        initial=dict(expected_generation=context['generation'],expected_revision=context['edit_revision'])
    reuse=current_scene==task['scene_id'] and context.get('weather','clear')==weather
    setup=add('scene','reset' if reuse else 'load_scene',initial if reuse else dict(initial,scene_id=task['scene_id'],weather=weather))
    def version(revision,source=setup):return dict(expected_generation={'$ref':source+'.result.generation'},expected_revision=revision)
    add('capabilities','get_capabilities',{});add('objects','list_objects',{})
    for i,frame in enumerate(task.get('verification_frames',[])):add(f'before_{i}','preview',dict(version(0),source_frame=frame,attitude_policy=policy))
    revision=0
    for i,command in enumerate(task.get('edits',[])):
        args=dict(version(revision),command=command)
        add(f'validate_{i}','validate_edit',args);add(f'edit_{i}','edit',args);revision+=1
    for i,preview in enumerate(task.get('previews',[])):
        args=copy.deepcopy(preview['args'])
        if preview['kind']=='ego_trajectory':
            args['trajectory']['attitude_policy']=policy;args['level_recorded_baseline']=policy=='gravity_level'
        else:args['attitude_policy']=policy
        add(f'preview_{i}',PREVIEWS[preview['kind']],dict(version(revision),**args))
    if 'ego_execution' in task:
        ego=copy.deepcopy({k:v for k,v in task['ego_execution'].items() if k!='driver'})
        ego['trajectory']['attitude_policy']=policy
        add('validate_drive','validate_ego_execution',dict(version(revision),**ego))
        add('drive','execute_ego_trajectory',dict(version(revision),**ego))
    if task.get('reset_after',True):
        reset=add('reset','reset',version(revision))
        for i,frame in enumerate(task.get('verification_frames',[])):add(f'after_{i}','preview',dict(version(0,reset),source_frame=frame,attitude_policy=policy))
    minimum=dict(scene_loads=0 if reuse else 1,resets=int(reuse)+int(task.get('reset_after',True)),edits=revision,validations=revision)
    if task.get('previews'):minimum['pre_run_previews']=sum(p['kind']=='camera' for p in task['previews'])
    if 'ego_execution' in task:minimum['ego_driving_executions']=1
    verification=dict(minimum_coverage=minimum,equal_rgb=[[names[f'before_{i}'],names[f'after_{i}']] for i in range(len(task.get('verification_frames',[])))])
    return validate_job(dict(schema='agent_job_v1',description=task.get('description','Unified scene task '+task['task_id']),steps=steps,verification=verification))


def _artifacts(job_output):
    result=[]
    for file in sorted(Path(job_output).glob('*index.json')):
        if file.name=='preview_index.json':result.append(dict(kind='observation_index',path=str(file)))
        for entry in _hugsim_expand(json.loads(file.read_text())):
            if file.name=='preview_index.json' and entry.get('action') in ('preview','observe'):
                result.append(dict(kind='observation',request_id=entry['request_id'],source_frame=entry['source_frame'],timestamp_s=entry['timestamp_s'],cameras=entry['cameras']))
            for key in ('video','manifest','closeup_video'):
                if entry.get(key):result.append(dict(kind=key,path=entry[key],source=file.name))
            comparison=entry.get('motion_comparison')
            if comparison:result.append(dict(kind='video',path=comparison['video'],source='motion_comparison'))
    return result


def execute_task(value,bridge,output,*,namespace=None):
    from agent_jobs import run_job
    initial_context=bridge.context()
    task=validate_task(value);job=compile_task(task,context=initial_context,namespace=namespace)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'task.json').write_text(json.dumps(task,indent=2));(out/'compiled_job.json').write_text(json.dumps(job,indent=2))
    weather_backend=bridge.session.backend;weather_start=len(getattr(weather_backend,'weather_records',[]))
    before=(bridge.render_count,bridge.control_count);error=None;acceptance=None;cleanup=None
    attitude_report=None
    try:
        acceptance=run_job(job,bridge,out/'job_results')
        from ego_attitude import audit_task_attitude
        attitude_report=audit_task_attitude(out/'job_results',task.get('attitude_policy','gravity_level'))
        acceptance['checks']['task_gravity_alignment']=attitude_report['passed']
        acceptance['check_applicability']['task_gravity_alignment']=attitude_report['applicable']
        if not attitude_report['applicable']:acceptance['not_exercised_checks'].append('task_gravity_alignment')
        acceptance['passed']=all(acceptance['checks'].values())
        (out/'job_results/acceptance.json').write_text(json.dumps(acceptance,indent=2))
        if not attitude_report['passed']:raise RuntimeError('task camera gravity alignment failed; inspect camera_attitude.json')
    except Exception as exc:
        path=out/'job_results/acceptance.json'
        if path.is_file():acceptance=_hugsim_expand(json.loads(path.read_text()))
        steps_path=out/'job_results/steps.json';steps=_hugsim_expand(json.loads(steps_path.read_text())) if steps_path.is_file() else []
        failed=next((s for s in steps if not s['matched']),None)
        error=dict(code='task_execution_failed',message=str(exc),failed_request_id=failed['request_id'] if failed else None,
            cause=failed['response'].get('error') if failed else None,failed_checks=[k for k,v in (acceptance or {}).get('checks',{}).items() if not v])
        setup_completed=bool(steps and steps[0]['response']['status']=='ok')
        if task.get('reset_after',True) and setup_completed and bridge.scene_id is not None:
            context=bridge.context()
            cleanup=bridge.handle(dict(schema='agent_request_v1',request_id='cleanup_'+hashlib.sha256(str(out).encode()).hexdigest()[:16],action='reset',args=dict(expected_generation=context['generation'],expected_revision=context['edit_revision'])))
    renders=bridge.render_count-before[0];controls=bridge.control_count-before[1]
    execution=dict(actual_renders=renders,executed_controls=controls,control_executed=controls>0,execution_backend=bridge.execution_backend,
        gpu_render_executed=bridge.execution_backend=='framework_renderer' and renders>0,
        driver='reference_tracking' if controls else None,feedback_source='native_motion_state' if controls else None,
        weather=task.get('weather','clear'),perception_connected=False,uniad_inference_executed=False,collision_checked=False,online_training=False)
    result=dict(schema='scene_task_result_v1',task_id=task['task_id'],status='completed' if error is None else 'failed',
        context=bridge.context(),execution=execution,artifacts=_artifacts(out/'job_results'),acceptance=acceptance,error=error,
        cleanup=cleanup,result_file=str(out/'task_result.json'),compiled_job=str(out/'compiled_job.json'),
        request_replay_scope='current owner process',transactional=False)
    backend=bridge.session.backend
    records=getattr(backend,'weather_records',[])
    records=records[weather_start:] if backend is weather_backend else records
    if records:
        weather_report=out/'weather_evidence.json';weather_report.write_text(json.dumps(dict(weather=task.get('weather','clear'),records=records),indent=2))
        result['artifacts'].append(dict(kind='weather_report',path=str(weather_report)))
        execution['weather_observations']=len(records)
        execution['weather_rgb_preservation_verified']=all(all(c['actor_rgb_preserved'] and c['depth_unchanged'] for c in record['cameras'].values()) for record in records)
    path=out/'job_results/acceptance.json'
    if path.is_file():result['artifacts'].append(dict(kind='acceptance',path=str(path)))
    metrics={}
    for filename,key,field in [('ego_driving_index.json','ego_tracking','tracking'),('actor_trajectory_index.json','actor_motion','motion_quality')]:
        index=out/'job_results'/filename
        if index.is_file():metrics[key]=[entry[field] for entry in _hugsim_expand(json.loads(index.read_text())) if entry.get(field) is not None]
    if attitude_report is not None:
        metrics['camera_horizon']=attitude_report
        result['artifacts'].append(dict(kind='camera_attitude',path=str(out/'job_results/camera_attitude.json')))
    result['metrics']=metrics
    if bridge.session.backend is not None:
        result['scene_summary']=bridge.session.spec.summary()
        try:result['scene_objects']=bridge.session.list_objects()
        except RuntimeError:result['scene_objects']=None
    result['recovery_required']=error is not None and not (cleanup and cleanup['status']=='ok') and bridge.context()!=initial_context
    (out/'task_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
