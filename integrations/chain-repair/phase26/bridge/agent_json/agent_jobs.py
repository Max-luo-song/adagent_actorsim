"""Finite JSON jobs emulate an editing agent using the same request API.

References only read earlier successful JSON responses; no expressions, shell,
code evaluation, implicit retries, or automatic fallback scenes/assets.
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

import copy
import hashlib
import json
pass  # Path is provided by the portable boundary above
from PIL import Image,ImageDraw
import numpy as np
from agent_bridge import BridgeError,fields,request
from agent_catalog import strict_json


def references(value):
    if isinstance(value,dict):
        if '$ref' in value:
            if set(value)!={'$ref'} or not isinstance(value['$ref'],str):raise ValueError('invalid response reference')
            yield value['$ref'];return
        for child in value.values():yield from references(child)
    elif isinstance(value,list):
        for child in value:yield from references(child)


def resolve(value,responses):
    if isinstance(value,dict):
        if '$ref' in value:
            if set(value)!={'$ref'} or not isinstance(value['$ref'],str):raise ValueError('invalid response reference')
            parts=value['$ref'].split('.')
            if len(parts)<2 or parts[0] not in responses:raise ValueError('unknown or forward response reference')
            current=responses[parts[0]]
            if current['status']!='ok':raise ValueError('cannot reference a failed request result')
            for part in parts[1:]:
                if isinstance(current,list) and part.isdigit():current=current[int(part)]
                elif isinstance(current,dict):current=current[part]
                else:raise ValueError('invalid reference path')
            return copy.deepcopy(current)
        return {k:resolve(v,responses) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v,responses) for v in value]
    return value


def validate_job(value,*,known_request_ids=()):
    fields(value,{'schema','steps'},{'description','verification'})
    if value['schema']!='agent_job_v1' or not isinstance(value['steps'],list) or not 1<=len(value['steps'])<=256:
        raise ValueError('agent_job_v1 requires 1..256 steps')
    previous=set();known=set(known_request_ids)
    for step in value['steps']:
        fields(step,{'request'},{'expect'})
        req=request(step['request'])
        if req['action']=='run_scene_task':raise ValueError('use scene_task_v1 or direct run_scene_task; nesting scene tasks in agent_job_v1 is unsupported')
        for ref in references(req):
            if ref.split('.')[0] not in previous:raise ValueError('unknown or forward reference: '+ref)
        seen_before=req['request_id'] in previous or req['request_id'] in known
        previous.add(req['request_id'])
        expect=step.get('expect',{'status':'ok'})
        fields(expect,{'status'},{'error_code','result','replayed'})
        if expect['status'] not in ('ok','error'):raise ValueError('expected status must be ok/error')
        if 'error_code' in expect and (expect['status']!='error' or not isinstance(expect['error_code'],str)):
            raise ValueError('error_code requires expected error status')
        if 'result' in expect and not isinstance(expect['result'],dict):raise ValueError('expected result must be an object subset')
        if 'replayed' in expect and type(expect['replayed']) is not bool:raise ValueError('expected replayed must be boolean')
        if expect.get('replayed') is True and not seen_before:
            raise ValueError('first occurrence of request_id cannot expect replayed=true in a fresh owner: '+req['request_id'])
    validate_verification(value)
    return copy.deepcopy(value)


def subset(actual,expected):
    if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and subset(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(actual)==len(expected) and all(subset(a,b) for a,b in zip(actual,expected))
    if type(expected) is bool:return type(actual) is bool and actual==expected
    return actual==expected


def matches(response,expect):
    return (response['status']==expect['status'] and
        ('error_code' not in expect or response.get('error',{}).get('code')==expect['error_code']) and
        ('result' not in expect or subset(response.get('result'),expect['result'])) and
        ('replayed' not in expect or response['replayed']==expect['replayed']))


def run_job(job,bridge,output):
    job=validate_job(job,known_request_ids=bridge.cache);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    responses={};steps=[];observations=[];trajectory_runs=[];driving_runs=[];actor_runs=[]
    initial_counts=(bridge.load_count,bridge.reset_count,bridge.render_count,bridge.control_count)
    try:
        for index,step in enumerate(job['steps']):
            req=resolve(step['request'],responses)
            response=bridge.handle(req);responses[req['request_id']]=response
            expect=step.get('expect',{'status':'ok'});ok=matches(response,expect)
            steps.append(dict(index=index,request_id=req['request_id'],action=req['action'],expected=expect,
                matched=ok,response=response))
            status=('expected rejection: '+response['error']['code']) if response['status']=='error' and ok else response['status']
            print(f"JSON request {index+1}/{len(job['steps'])}: {req['action']} -> {status}; replayed={response['replayed']}; matched={ok}",flush=True)
            if response['status']=='ok' and req['action'] in ('preview','observe') and not response['replayed']:observations.append(dict(response['result'],request_id=req['request_id'],action=req['action']))
            if response['status']=='ok' and req['action']=='preview_ego_trajectory' and not response['replayed']:
                result=response['result'];manifest=_hugsim_expand(json.loads(Path(result['manifest']).read_text()))
                trajectory_runs.append(dict(request_id=req['request_id'],**result))
                for index,pair in enumerate(manifest['frames']):
                    for kind,packet in pair.items():
                        if packet is not None:observations.append(dict(packet,request_id=f'{req["request_id"]}_{index:04d}_{kind}',action='preview_ego_trajectory'))
            if response['status']=='ok' and req['action']=='preview_actor_trajectory' and not response['replayed']:
                result=response['result'];manifest=_hugsim_expand(json.loads(Path(result['manifest']).read_text()))
                actor_runs.append(dict(request_id=req['request_id'],**result))
                for index,pair in enumerate(manifest['frames']):
                    for kind in ('baseline','new'):
                        observations.append(dict(pair[kind],request_id=f'{req["request_id"]}_{index:04d}_{kind}',action='preview_actor_trajectory'))
            if response['status']=='ok' and req['action']=='execute_ego_trajectory' and not response['replayed']:
                result=response['result'];manifest=_hugsim_expand(json.loads(Path(result['manifest']).read_text()))
                driving_runs.append(dict(request_id=req['request_id'],**result))
                for index,packet in enumerate(manifest['frames']):
                    observations.append(dict(packet,request_id=f'{req["request_id"]}_{index:04d}_executed',action='execute_ego_trajectory'))
            if not ok:break
    finally:
        checks=dict(all_steps_executed=len(steps)==len(job['steps']),
            expected_results_matched=bool(steps) and all(s['matched'] for s in steps),
            editable_previews_stay_pre_run=all(s['response']['context']['phase']=='pre_run' for s in steps if s['action']=='preview' and s['response']['status']=='ok' and not s['response']['replayed']),
            validation_preserves_scene_revision=all(s['response']['result']['mutation_performed'] is False and s['response']['context']['edit_revision']+1==s['response']['result']['would_edit_revision'] for s in steps if s['action']=='validate_edit' and s['response']['status']=='ok' and not s['response']['replayed']),
            edits_commit_one_revision=all(s['response']['result']['receipt']['edit_revision']==s['response']['result']['receipt']['previous_revision']+1 for s in steps if s['action']=='edit' and s['response']['status']=='ok'),
            successful_resets_verified=all(all(s['response']['result']['reset_audit'].values()) for s in steps if s['action']=='reset' and s['response']['status']=='ok'),
            no_online_training=all(s['response']['training_performed'] is False for s in steps),
            no_control_misrepresentation=all(
                (s['response']['control_executed']==(s['response'].get('controls_executed',0)>0)) and
                (not s['response']['control_executed'] or s['action']=='execute_ego_trajectory') for s in steps),
            preview_files_valid=all(Path(c['rgb']).is_file() and Path(c['depth']).is_file() and
                hashlib.sha256(np.load(c['depth']).tobytes()).hexdigest()==c['depth_sha256'] and
                hashlib.sha256(np.array(Image.open(c['rgb'])).tobytes()).hexdigest()==c['rgb_sha256']
                for o in observations for c in o['cameras'].values()))
        coverage=execution_coverage(steps)
        verification=verify_evidence(job.get('verification',{}),coverage,steps,observations)
        checks.update(verification['checks'])
        checks['weather_evidence_verified']=all(o['environment']['actor_rgb_preserved'] and o['environment']['depth_unchanged'] for o in observations if o.get('environment',{}).get('weather')=='rainy')
        checks['ego_trajectory_previews_verified']=all(
            Path(run['video']).is_file() and Path(run['video']).stat().st_size>0 and
            _hugsim_expand(json.loads(Path(run['manifest']).read_text()))['completed'] and
            all(run['audit'].values()) and run['preview_only'] and not run['observation_history_committed'] and
            run['rendered_packets']==len(_hugsim_expand(json.loads(Path(run['manifest']).read_text()))['frames'])*(2 if _hugsim_expand(json.loads(Path(run['manifest']).read_text()))['frames'][0]['baseline'] else 1)
            for run in trajectory_runs)
        checks['ego_trajectory_preserves_edit_phase']=all(
            s['response']['context']['phase']=='pre_run' and
            s['response']['context']['generation']==s['response']['result']['generation'] and
            s['response']['context']['edit_revision']==s['response']['result']['edit_revision']
            for s in steps if s['action']=='preview_ego_trajectory' and s['response']['status']=='ok' and not s['response']['replayed'])
        checks['ego_execution_verified']=all(all(run['audit'].values()) and
            _hugsim_expand(json.loads(Path(run['manifest']).read_text()))['completed'] and Path(run['video']).is_file() and
            run['control_executed'] and not run['preview_only'] and run['executed_controls']>0
            for run in driving_runs)
        checks['ego_execution_tracking_within_tolerance']=all(run['tracking']['passed'] for run in driving_runs)
        checks['control_receipts_match_execution']=bridge.control_count-initial_counts[3]==sum(
            s['response'].get('controls_executed',0) for s in steps if not s['response']['replayed'])
        from ego_attitude import present_attitude_comparison
        comparison=present_attitude_comparison(trajectory_runs,out,bridge.execution_backend)
        checks['ego_attitude_comparison_passed']=comparison['passed'] if comparison is not None else True
        from traffic_presentation import present_signals
        signal_report=present_signals(observations,out,bridge.execution_backend)
        checks['signal_color_probe_passed']=signal_report['passed'] if signal_report is not None else True
        checks['actor_trajectory_previews_verified']=all(all(run['audit'].values()) and Path(run['closeup_video']).is_file() and
            _hugsim_expand(json.loads(Path(run['manifest']).read_text()))['completed'] and Path(run['video']).is_file() and
            run['preview_only'] and not run['observation_history_committed'] and
            run['rendered_packets']==2*len(_hugsim_expand(json.loads(Path(run['manifest']).read_text()))['frames']) for run in actor_runs)
        checks['actor_motion_comparison_verified']=all(run['motion_comparison']['passed'] and Path(run['motion_comparison']['video']).is_file() for run in actor_runs if run.get('motion_comparison'))
        from ego_attitude import compare_camera_levels
        camera_comparison=compare_camera_levels(observations,out,bridge.execution_backend)
        checks['ordinary_camera_alignment_comparison']=camera_comparison['passed'] if camera_comparison else True
        applicable={key:True for key in checks}
        applicable['ordinary_camera_alignment_comparison']=camera_comparison is not None
        applicable['weather_evidence_verified']=any(o.get('environment',{}).get('weather')=='rainy' for o in observations)
        applicable['actor_trajectory_previews_verified']=bool(actor_runs)
        applicable['actor_motion_comparison_verified']=any(run.get('motion_comparison') for run in actor_runs)
        applicable.update(signal_color_probe_passed=signal_report is not None,ego_attitude_comparison_passed=comparison is not None,ego_execution_verified=bool(driving_runs),ego_execution_tracking_within_tolerance=bool(driving_runs),control_receipts_match_execution=bridge.control_count>initial_counts[3],ego_trajectory_previews_verified=bool(trajectory_runs),ego_trajectory_preserves_edit_phase=bool(trajectory_runs),editable_previews_stay_pre_run=coverage['pre_run_previews']>0,
            validation_preserves_scene_revision=coverage['validations']>0,
            edits_commit_one_revision=coverage['edits']>0,
            successful_resets_verified=coverage['resets']>0,
            preview_files_valid=len(observations)>0,
            required_coverage_met=bool(job.get('verification',{}).get('minimum_coverage')),
            required_rejections_observed=bool(job.get('verification',{}).get('required_rejections')),
            declared_rgb_comparisons_passed=bool(job.get('verification',{}).get('equal_rgb') or job.get('verification',{}).get('different_rgb')))
        acceptance=dict(passed=all(checks.values()),coverage=coverage,check_applicability=applicable,
            not_exercised_checks=[key for key,value in applicable.items() if not value],checks=checks,requested_steps=len(job['steps']),completed_steps=len(steps),
            expected_error_count=sum(s['response']['status']=='error' and s['matched'] for s in steps),
            actor_trajectory_previews=len(actor_runs),ego_driving_executions=len(driving_runs),executed_controls=bridge.control_count-initial_counts[3],
            ego_trajectory_previews=len(trajectory_runs),rendered_packets=len(observations),pre_run_previews=sum(o['preview_only'] for o in observations),
            committed_observations=sum(not o['preview_only'] for o in observations),scene_loads=bridge.load_count-initial_counts[0],scene_resets=bridge.reset_count-initial_counts[1],
            actual_renders=bridge.render_count-initial_counts[2],execution_backend=bridge.execution_backend,
            gpu_render_executed=bridge.execution_backend=='framework_renderer' and bridge.render_count-initial_counts[2]>0,
            closed_loop_driving=bridge.control_count>initial_counts[3],feedback_source='native_motion_state' if bridge.control_count>initial_counts[3] else None,
            perception_connected=False,uniad_inference_executed=False,collision_checked=False,online_training=False,visual_quality_certified=False,formal_evaluation_admitted=False)
        (out/'actor_trajectory_index.json').write_text(json.dumps(actor_runs,indent=2,allow_nan=False))
        (out/'ego_driving_index.json').write_text(json.dumps(driving_runs,indent=2,allow_nan=False))
        (out/'ego_trajectory_index.json').write_text(json.dumps(trajectory_runs,indent=2,allow_nan=False))
        (out/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False))
        (out/'preview_index.json').write_text(json.dumps(observations,indent=2,allow_nan=False))
        (out/'steps.json').write_text(json.dumps(steps,indent=2,allow_nan=False))
        (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
    if observations:present_previews([o for o in observations if o['action'] in ('preview','observe')] or observations[:2],out,bridge.execution_backend)
    if not acceptance['passed']:raise RuntimeError('agent JSON job failed; inspect steps.json and acceptance.json')
    return acceptance


def present_previews(observations,output,backend):
    """Contact sheet from actual saved camera PNGs; no interpolated frames."""
    previews=[o for o in observations if o['preview_only']] or observations
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
    width=360;height=240;header=32
    sheet=Image.new('RGB',(width*3,(height+header)*len(previews)+32),(20,20,20))
    draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'JSON observations | '+backend+' | '+('executed state-feedback driving' if any(o['action']=='execute_ego_trajectory' for o in observations) else 'no driving'),fill='white')
    for row,packet in enumerate(previews):
        y=32+row*(height+header)
        label=f"{packet['request_id']} | scene={packet['scene']} frame={packet['source_frame']} revision={packet['edit_revision']}"
        draw.text((8,y+7),label,fill='white')
        for col,name in enumerate(order):
            im=Image.open(packet['cameras'][name]['rgb']).convert('RGB');im.thumbnail((width,height))
            sheet.paste(im,(col*width,y+header))
    sheet.save(Path(output)/'preview_comparison.jpg',quality=94)


COVERAGE_ACTIONS={'actor_trajectory_previews':'preview_actor_trajectory','actor_trajectory_edits':'edit','scene_loads':'load_scene','edits':'edit','native_edits':'edit','managed_edits':'edit','calibrated_signal_edits':'edit',
    'ego_driving_executions':'execute_ego_trajectory','ego_execution_validations':'validate_ego_execution',
    'ego_trajectory_previews':'preview_ego_trajectory','ego_trajectory_validations':'validate_ego_trajectory',
    'resets':'reset','validations':'validate_edit','pre_run_previews':'preview',
    'committed_observations':'observe','replayed_requests':None,'rejected_requests':None}


def validate_verification(job):
    spec=job.get('verification',{})
    fields(spec,set(),{'minimum_coverage','required_rejections','equal_rgb','different_rgb'})
    minimum=spec.get('minimum_coverage',{})
    if not isinstance(minimum,dict) or set(minimum)-set(COVERAGE_ACTIONS):raise ValueError('unknown coverage requirement')
    for key,value in minimum.items():
        if type(value) is not int or value<0:raise ValueError('minimum coverage must be a nonnegative integer')
        action=COVERAGE_ACTIONS[key]
        upper=sum(step['request']['action']==action for step in job['steps']) if action else len(job['steps'])
        if value>upper:raise ValueError('coverage requirement exceeds declared actions: '+key)
    required=spec.get('required_rejections',[])
    if not isinstance(required,list) or any(not isinstance(x,str) or not x for x in required):raise ValueError('required_rejections must contain nonempty error codes')
    actions={step['request']['request_id']:step['request']['action'] for step in job['steps']}
    for kind in ('equal_rgb','different_rgb'):
        pairs=spec.get(kind,[])
        if not isinstance(pairs,list):raise ValueError(kind+' must be a list of request-ID pairs')
        for pair in pairs:
            if (not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(x,str) for x in pair)
                    or pair[0]==pair[1] or any(actions.get(x) not in ('preview','observe') for x in pair)):
                raise ValueError('RGB comparison requires two distinct declared preview/observe request IDs')


def execution_coverage(steps):
    fresh=[s for s in steps if s['response']['status']=='ok' and not s['response']['replayed']]
    edits=[s for s in fresh if s['action']=='edit']
    return dict(actor_trajectory_previews=sum(s['action']=='preview_actor_trajectory' for s in fresh),
        actor_trajectory_edits=sum(s['response']['result']['receipt']['command']['operation'] in ('set_actor_trajectory','restore_actor_trajectory') for s in edits),ego_driving_executions=sum(s['action']=='execute_ego_trajectory' for s in fresh),
        ego_execution_validations=sum(s['action']=='validate_ego_execution' for s in fresh),ego_trajectory_previews=sum(s['action']=='preview_ego_trajectory' for s in fresh),
        ego_trajectory_validations=sum(s['action']=='validate_ego_trajectory' for s in fresh),scene_loads=sum(s['action']=='load_scene' for s in fresh),resets=sum(s['action']=='reset' for s in fresh),
        edits=len(edits),calibrated_signal_edits=sum(s['response']['result']['receipt']['command']['object_id'].startswith('signal:') for s in edits),native_edits=sum(s['response']['result']['receipt']['command']['object_id'].startswith('native:') for s in edits),
        managed_edits=sum(s['response']['result']['receipt']['command']['object_id'].startswith('managed:') for s in edits),
        validations=sum(s['action']=='validate_edit' for s in fresh),
        pre_run_previews=sum(s['action']=='preview' for s in fresh),committed_observations=sum(s['action']=='observe' for s in fresh),
        replayed_requests=sum(s['response']['replayed'] for s in steps),
        rejected_requests=sum(s['response']['status']=='error' for s in steps))


def verify_evidence(spec,coverage,steps,observations):
    met={key:dict(actual=coverage[key],minimum=value,passed=coverage[key]>=value) for key,value in spec.get('minimum_coverage',{}).items()}
    codes=sorted({s['response']['error']['code'] for s in steps if s['response']['status']=='error'})
    rejections={key:key in codes for key in spec.get('required_rejections',[])}
    by_id={o['request_id']:o for o in observations};comparisons=[]
    for kind in ('equal_rgb','different_rgb'):
        for first,second in spec.get(kind,[]):
            result=dict(relation=kind,first_request=first,second_request=second,passed=False)
            try:
                a,b=by_id[first],by_id[second]
                comparable=(a['scene']==b['scene'] and a['source_frame']==b['source_frame'] and
                    a['timestamp_s']==b['timestamp_s'] and np.allclose(a['ego_to_world'],b['ego_to_world'],atol=1e-9,rtol=0) and
                    set(a['cameras'])==set(b['cameras']))
                pixels={}
                for cam,source in a['cameras'].items():
                    target=b['cameras'][cam]
                    comparable &= np.allclose(source['K'],target['K'],atol=1e-9,rtol=0) and np.allclose(source['camera_to_world'],target['camera_to_world'],atol=1e-9,rtol=0)
                    x=np.array(Image.open(source['rgb']));y=np.array(Image.open(target['rgb']))
                    if x.shape!=y.shape:raise ValueError('camera image shapes differ')
                    pixels[cam]=int(np.any(x!=y,axis=-1).sum())
                result.update(comparable=bool(comparable),rgb_changed_pixels=pixels,
                    passed=bool(comparable and (all(v==0 for v in pixels.values()) if kind=='equal_rgb' else any(v>0 for v in pixels.values()))))
            except (KeyError,OSError,ValueError) as exc:result['error']=str(exc)
            comparisons.append(result)
    checks=dict(required_coverage_met=all(v['passed'] for v in met.values()),
        required_rejections_observed=all(rejections.values()),declared_rgb_comparisons_passed=all(c['passed'] for c in comparisons))
    return dict(checks=checks,coverage=coverage,coverage_requirements=met,observed_rejection_codes=codes,
        required_rejections=rejections,comparisons=comparisons,comparisons_use_saved_png=True,
        visual_quality_certified=False,closed_loop_driving=False)
