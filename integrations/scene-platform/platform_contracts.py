"""Portable editing/advance instructions; no caller-supplied code or model parameters."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import copy,json
from platform_io import identity
from sensor_contracts import SCHEMA as SENSOR_SCHEMA,configuration as sensor_configuration
SCHEMA={
 '$schema':'https://json-schema.org/draft/2020-12/schema','type':'object','additionalProperties':False,
 'required':['schema','task_id','scene_id','edits','simulation','reset_after'],
 'properties':{
  'schema':{'const':'scene_task_v2'},'task_id':{'type':'string','pattern':'^[A-Za-z][A-Za-z0-9_-]{0,63}$'},
  'scene_id':{'type':'string'},'description':{'type':'string'},'weather':{'const':'clear'},
  'edits':{'type':'array','maxItems':1,'items':{'type':'object','additionalProperties':False,
    'required':['operation','object_id','program_id'],'properties':{'operation':{'const':'set_actor_trajectory'},'object_id':{'type':'string'},'program_id':{'type':'string'}}}},
  'simulation':{'type':'object','additionalProperties':False,'required':['driver','profile','control_steps'],
    'properties':{'driver':{'const':'uniad_closed_loop'},'profile':{'type':'string'},'control_steps':{'type':'integer','minimum':1,'maximum':188}}},
  'reset_after':{'const':True},'evaluation_mode':{'const':'diagnostic'},
  'sensors':{'type':'object','additionalProperties':False,'properties':{'lidar':SENSOR_SCHEMA}}
 }}

def fields(v,required,optional=()):
    if not isinstance(v,dict) or not set(required)<=set(v) or set(v)-set(required)-set(optional):raise ValueError('Invalid fields: expected '+str(sorted(required)))

def validate(task,catalog):
    task=_hugsim_expand(json.loads(json.dumps(task,allow_nan=False)));fields(task,['schema','task_id','scene_id','edits','simulation','reset_after'],['description','weather','evaluation_mode','sensors'])
    if task['schema']!='scene_task_v2':raise ValueError('Expected scene_task_v2')
    identity(task['task_id'])
    if task.get('weather','clear')!='clear':raise ValueError('Closed-loop weather combination not admitted yet')
    if task.get('evaluation_mode','diagnostic')!='diagnostic':raise ValueError('Only explicit diagnostic execution is registered')
    if task['reset_after'] is not True:raise ValueError('This episode adapter requires verified reset and close')
    if 'description' in task and not isinstance(task['description'],str):raise ValueError('Description must be text')
    sensors=task.get('sensors',{});fields(sensors,[],['lidar']);sensor_configuration(sensors.get('lidar'))
    sim=task['simulation'];fields(sim,['driver','profile','control_steps'])
    if sim['driver']!='uniad_closed_loop':raise ValueError('Unsupported driver; use legacy scene_task_v1 for reference tracking')
    profile=catalog.profile(sim['profile'])
    if task['scene_id']!=profile['scene_id']:raise ValueError('Profile belongs to another scene')
    if type(sim['control_steps']) is not int or not 1<=sim['control_steps']<=profile['max_control_steps']:raise ValueError('Invalid control budget')
    if not isinstance(task['edits'],list) or len(task['edits'])>1:raise ValueError('First closed-loop adapter supports one registered native trajectory or no edit')
    for edit in task['edits']:
        fields(edit,['operation','object_id','program_id'])
        if edit['operation']!='set_actor_trajectory':raise ValueError('Unsupported edit operation')
        program=catalog.program(edit['program_id'])
        if program['scene_id']!=task['scene_id'] or program['object_id']!=edit['object_id'] or program['driver_profile']!=sim['profile']:raise ValueError('Program/object/scene/profile binding mismatch')
    task.setdefault('weather','clear');task.setdefault('evaluation_mode','diagnostic')
    return task

def compile_task(task,catalog):
    if isinstance(task,dict) and task.get('schema')=='scene_task_v1':
        # Full existing validators run in the isolated worker before any GPU work.
        from platform_io import read
        legacy=catalog.data['legacy'];schema=read(legacy['task_schema']['path'])
        task=_hugsim_expand(json.loads(json.dumps(task,allow_nan=False)))
        fields(task,schema['required'],set(schema['properties'])-set(schema['required']))
        identity(task['task_id'])
        if not isinstance(task['scene_id'],str):raise ValueError('Invalid legacy scene ID')
        return dict(schema='compiled_scene_task_v2',task=task,profile_id='legacy_scene_task_v1',
            module_calls=[dict(module='legacy_scene_task_adapter',operation='existing_validate_compile_execute',
                task_schema='scene_task_v1')],LLM_invoked=False,
            validation_scope='Envelope only; authoritative existing module preflight required before execution')
    task=validate(task,catalog);profile=catalog.profile(task['simulation']['profile'])
    modules=[dict(module='scene_loader',operation='load_scene',scene_id=task['scene_id'])]
    modules.extend(dict(module='native_vehicle_trajectory',**e) for e in task['edits'])
    sensors=sensor_configuration(task.get('sensors',{}).get('lidar'))
    if sensors['enabled']:modules.append(dict(module='optional_pointcloud_recorder',operation='record_only',configuration=sensors))
    modules.extend([dict(module='closed_loop_driver',operation='advance',**task['simulation']),
        dict(module='session_lifecycle',operation='reset_and_verify_then_close'),
        dict(module='native_evaluator',operation='score_plans_and_actual_execution'),
        dict(module='presentation',operation='render_execution_video')])
    return dict(schema='compiled_scene_task_v2',task=task,profile_id=task['simulation']['profile'],
        arm='B' if task['edits'] else 'A',control_steps=task['simulation']['control_steps'],
        module_calls=modules,source_start_frame=profile['source_start_frame'],dt_s=profile['dt_s'],
        source_end_frame=profile['source_start_frame']+task['simulation']['control_steps'],
        source_time_exhaustion_must_not_be_clamped=True,LLM_invoked=False)

def tool_specs():
    from platform_io import P26,read
    legacy=read(P26/'bridge/agent_json/schemas/scene_task_v1.schema.json')
    return [dict(name='get_scene_capabilities',description='Query registered scenes, edit programs and exact closed-loop driver versions.',parameters={'type':'object','properties':{},'additionalProperties':False}),
        dict(name='run_scene_task',description='Dispatch existing scene_task_v1 editing/preview/reference execution, or scene_task_v2 actual UniAD closed-loop driving. Read capabilities before combining modules.',parameters={'type':'object','required':['task'],'additionalProperties':False,'properties':{'task':{'oneOf':[legacy,SCHEMA]}}})]

