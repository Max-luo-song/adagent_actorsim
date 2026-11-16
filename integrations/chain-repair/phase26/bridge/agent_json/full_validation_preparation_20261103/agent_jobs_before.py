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
    fields(value,{'schema','steps'},{'description'})
    if value['schema']!='agent_job_v1' or not isinstance(value['steps'],list) or not 1<=len(value['steps'])<=256:
        raise ValueError('agent_job_v1 requires 1..256 steps')
    previous=set();known=set(known_request_ids)
    for step in value['steps']:
        fields(step,{'request'},{'expect'})
        req=request(step['request'])
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
    responses={};steps=[];observations=[]
    initial_counts=(bridge.load_count,bridge.reset_count,bridge.render_count)
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
            if not ok:break
    finally:
        checks=dict(all_steps_executed=len(steps)==len(job['steps']),
            expected_results_matched=bool(steps) and all(s['matched'] for s in steps),
            editable_previews_stay_pre_run=all(s['response']['context']['phase']=='pre_run' for s in steps if s['action']=='preview' and s['response']['status']=='ok' and not s['response']['replayed']),
            validation_preserves_scene_revision=all(s['response']['result']['mutation_performed'] is False and s['response']['context']['edit_revision']+1==s['response']['result']['would_edit_revision'] for s in steps if s['action']=='validate_edit' and s['response']['status']=='ok' and not s['response']['replayed']),
            edits_commit_one_revision=all(s['response']['result']['receipt']['edit_revision']==s['response']['result']['receipt']['previous_revision']+1 for s in steps if s['action']=='edit' and s['response']['status']=='ok'),
            successful_resets_verified=all(all(s['response']['result']['reset_audit'].values()) for s in steps if s['action']=='reset' and s['response']['status']=='ok'),
            no_online_training=all(s['response']['training_performed'] is False for s in steps),
            no_control_misrepresentation=all(s['response']['control_executed'] is False for s in steps),
            preview_files_valid=all(Path(c['rgb']).is_file() and Path(c['depth']).is_file() and
                hashlib.sha256(np.load(c['depth']).tobytes()).hexdigest()==c['depth_sha256'] and
                hashlib.sha256(np.array(Image.open(c['rgb'])).tobytes()).hexdigest()==c['rgb_sha256']
                for o in observations for c in o['cameras'].values()))
        acceptance=dict(passed=all(checks.values()),checks=checks,requested_steps=len(job['steps']),completed_steps=len(steps),
            expected_error_count=sum(s['response']['status']=='error' and s['matched'] for s in steps),
            rendered_packets=len(observations),pre_run_previews=sum(o['preview_only'] for o in observations),
            committed_observations=sum(not o['preview_only'] for o in observations),scene_loads=bridge.load_count-initial_counts[0],scene_resets=bridge.reset_count-initial_counts[1],
            actual_renders=bridge.render_count-initial_counts[2],execution_backend=bridge.execution_backend,
            gpu_render_executed=bridge.execution_backend=='framework_renderer' and bridge.render_count-initial_counts[2]>0,
            closed_loop_driving=False,online_training=False,visual_quality_certified=False,formal_evaluation_admitted=False)
        (out/'preview_index.json').write_text(json.dumps(observations,indent=2,allow_nan=False))
        (out/'steps.json').write_text(json.dumps(steps,indent=2,allow_nan=False))
        (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
    if observations:present_previews(observations,out,bridge.execution_backend)
    if not acceptance['passed']:raise RuntimeError('agent JSON job failed; inspect steps.json and acceptance.json')
    return acceptance


def present_previews(observations,output,backend):
    """Contact sheet from actual saved camera PNGs; no interpolated frames."""
    previews=[o for o in observations if o['preview_only']] or observations
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
    width=360;height=240;header=32
    sheet=Image.new('RGB',(width*3,(height+header)*len(previews)+32),(20,20,20))
    draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'JSON editing preview | '+backend+' | no driving',fill='white')
    for row,packet in enumerate(previews):
        y=32+row*(height+header)
        label=f"{packet['request_id']} | scene={packet['scene']} frame={packet['source_frame']} revision={packet['edit_revision']}"
        draw.text((8,y+7),label,fill='white')
        for col,name in enumerate(order):
            im=Image.open(packet['cameras'][name]['rgb']).convert('RGB');im.thumbnail((width,height))
            sheet.paste(im,(col*width,y+header))
    sheet.save(Path(output)/'preview_comparison.jpg',quality=94)
