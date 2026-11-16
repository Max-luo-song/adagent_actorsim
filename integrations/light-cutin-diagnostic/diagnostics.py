"""Preregistered event definitions, first-divergence diagnosis and paired artifacts."""

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
import pickle
pass  # Path is provided by the portable boundary above
import numpy as np
from common import read,write,asset
from runtime import TERMINATION

def target_association(target_source,info,arrays):
    """Post-hoc current-truth association only; never passed to reference selection."""
    center=np.asarray(target_source,dtype=float)
    virtual_world=np.array([-center[1],center[0],center[2]])
    local=(virtual_world-np.asarray(info['explicit_l2g_t']))@np.asarray(info['explicit_l2g_r_mat'])
    result=dict(current_target_virtual_xyz=local.tolist(),reference_only=True,used_by_online_policy=False)
    for role,box_key,score_key in [('retained','boxes','scores'),('candidate','geometry_candidate_boxes','candidate_scores'),('motion_query','motion_query_boxes','motion_query_scores')]:
        if box_key not in arrays:continue
        boxes=np.asarray(arrays[box_key]);scores=np.asarray(arrays.get(score_key,[]))
        if role=='motion_query':boxes=boxes[:-1];scores=scores[:-1]  # original motion head's last query is SDC
        if len(boxes)==0:result[role]=None;continue
        distances=np.linalg.norm(boxes[:,:2]-local[:2],axis=1);index=int(np.argmin(distances))
        result[role]=dict(index=index,xy_distance_m=float(distances[index]),score=float(scores[index]) if len(scores)>index else None,
            box=boxes[index].tolist(),association='nearest current center; ambiguity remains, not a certified track identity')
    return result

def arm_summary(path):
    path=Path(path);r=read(path/'run_report.json');trace=r['trace'];obs=r['observations']
    speeds=np.array([s['motion_state'][3] for s in obs]);positions=np.array([s['pose'] for s in obs])[:,:2,3] if obs else np.empty((0,2))
    stopped=[];recoveries=[];start=None;recover_start=None;abnormal=[]
    for i,o in enumerate(obs):
        if speeds[i]<=TERMINATION['stop_speed_mps']:
            if start is None:start=i
            recover_start=None
            duration=(i-start)*.1
            if duration>=TERMINATION['stagnation_hold_s'] and np.linalg.norm(positions[i]-positions[start])<=TERMINATION['stagnation_displacement_m']:
                recent=[t['control']['target_speed_mps'] for t in trace[max(0,i-10):i]]
                if len(recent)>=10 and min(recent)>=TERMINATION['abnormal_stall_reference_speed_mps']:abnormal.append(i)
        elif start is not None:
            if speeds[i]>TERMINATION['recovery_speed_mps']:
                if recover_start is None:recover_start=i
                if (i-recover_start)*.1>=TERMINATION['recovery_hold_s']:
                    stopped.append(dict(start=start,end=recover_start,hold_s=(recover_start-start)*.1));recoveries.append(dict(stop_start=start,recovered_at=i));start=None;recover_start=None
            else:recover_start=None
    if start is not None:stopped.append(dict(start=start,end=None,hold_s=(len(obs)-1-start)*.1))
    collisions=bool(r.get('terminal_collision')) or any(o['environment']['collision'] for o in obs)
    route=any(o['environment']['native_route_deviation'] for o in obs)
    max_error=max((t['tracking_error']['position_m'] for t in trace),default=None)
    native_path=path/'native_eval/native_scores.json';native=read(native_path) if native_path.exists() else None
    executed=native.get('scores',{}).get('executed_path') if native and native.get('scores') else None
    associations=[]
    for row in obs:
        index=row['index'];target=row['environment'].get('target_world');folder=path/f'uniad_worker/step_{index:03d}'
        if target is None or not (folder/'perception.npz').exists():continue
        with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
        with np.load(folder/'perception.npz') as z:arrays={k:z[k].copy() for k in z.files}
        if (folder/'motion_head_raw.npz').exists():
            with np.load(folder/'motion_head_raw.npz') as z:
                for k in ['motion_query_boxes','motion_query_scores']:
                    if k in z:arrays[k]=z[k].copy()
        associations.append(dict(index=index,**target_association(target['box'][:3],info,arrays)))
    write(path/'target_detection_association_reference_only.json',dict(rows=associations,used_by_online_policy=False))
    proved_native_dac_failure=bool(executed and executed['dac']['upper']<.9)
    anomalies=dict(environment_collision=collisions,native_route_deviation=route,abnormal_stall=bool(abnormal),
        known_executed_native_dac_below_09=proved_native_dac_failure,tracker_error_over_1m=max_error is not None and max_error>1.,runtime_error=r['status']=='failed')
    collision_times=[o['timestamp'] for o in obs if o['environment']['collision']]
    if r.get('terminal_collision',{}).get('timestamp_s') is not None:collision_times.append(r['terminal_collision']['timestamp_s'])
    first_anomalies=dict(environment_collision=min(collision_times) if collision_times else None,
        native_route_deviation=next((o['timestamp'] for o in obs if o['environment']['native_route_deviation']),None),
        tracking_error_over_1m=next(((t['index']+1)*.1 for t in trace if t['tracking_error']['position_m']>1.),None),
        abnormal_stall=min(abnormal)*.1 if abnormal else None,
        runtime_error=r.get('last_step_index',0)*.1 if r['status']=='failed' else None)
    result=dict(status=r['status'],termination_reason=r.get('termination_reason'),controls=len(trace),observations=len(obs),
        elapsed_simulation_s=obs[-1]['timestamp'] if obs else 0.,wall_seconds=r['wall_seconds'],
        anomalies=anomalies,any_anomaly=any(anomalies.values()),first_anomaly_time_s=first_anomalies,
        error_stage=r.get('current_stage') if r['status']=='failed' else None,
        stops=stopped,recoveries=recoveries,abnormal_stall_steps=abnormal,
        no_recovery_is_not_failure_if_observation_ends=True,max_one_step_tracking_error_m=max_error,
        stopped_seconds=float(np.sum(speeds<=TERMINATION['stop_speed_mps']))*.1,
        stop_recovery_unresolved=bool(stopped and stopped[-1]['end'] is None and stopped[-1]['hold_s']>=TERMINATION['stop_hold_s']),
        mean_tracking_solve_seconds=float(np.mean([t['control']['solve_seconds'] for t in trace])) if trace else None,
        rendering_seconds=sum(t.get('render_seconds',0) for t in trace),
        inference_seconds=sum(t.get('inference_seconds') or 0 for t in trace),startup_timing=r.get('timing',{}),
        external_reference_changes=sum(t.get('external_reference_changed',False) for t in trace),
        progress_along_initial_forward_m=float(positions[-1,0]-positions[0,0]) if len(positions) else None,
        actual_hd_road_departure=None,physical_road_boundary_verified=False,
        native_scores=native,executed_score_is_distinct_from_raw_adopted_plans=True,run=asset(path/'run_report.json'))
    write(path/'summary.json',result);return result


def paired(root,left='A',right='B'):
    root=Path(root);a=read(root/left/'run_report.json');b=read(root/right/'run_report.json')
    role='scene_intervention' if left=='A' and right in ('B','B_repeat1') else 'external_patch'
    actor_tracks_identical=None
    if role=='external_patch':
        with np.load(root/left/'actor_tracks.npz') as x,np.load(root/right/'actor_tracks.npz') as y:
            actor_tracks_identical=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        if not actor_tracks_identical:raise ValueError('Patch pair changed actor trajectories; not a one-factor patch comparison')
    aa={r['index']:r for r in a['trace']};bb={r['index']:r for r in b['trace']}
    ao={r['index']:r for r in a['observations']};bo={r['index']:r for r in b['observations']}
    first={k:None for k in ['camera_pixels','detection','motion_prediction','raw_plan','adopted_reference','control','executed_state']};rows=[]
    for index in sorted(set(ao)&set(bo)):
        x,y=ao[index],bo[index];different={}
        different['camera_pixels']=x['image_sha256']!=y['image_sha256']
        different['executed_state']=not np.allclose(x['motion_state'],y['motion_state'],rtol=0,atol=1e-8)
        for key,field in [('raw_plan','raw_uniad'),('adopted_reference','adopted')]:
            different[key]=index in aa and index in bb and not np.allclose(aa[index][field],bb[index][field],rtol=0,atol=1e-4)
        different['control']=index in aa and index in bb and not np.allclose(list(aa[index]['action'].values()),list(bb[index]['action'].values()),rtol=0,atol=1e-5)
        p=root/left/f'uniad_worker/step_{index:03d}/perception.npz';q=root/right/f'uniad_worker/step_{index:03d}/perception.npz'
        if p.exists() and q.exists():
            with np.load(p) as u,np.load(q) as v:
                keys=[k for k in ['boxes','scores','labels'] if k in u and k in v]
                different['detection']=any(u[k].shape!=v[k].shape or not np.allclose(u[k],v[k],rtol=0,atol=1e-4) for k in keys)
        p=root/left/f'uniad_worker/step_{index:03d}/motion_head_raw.npz';q=root/right/f'uniad_worker/step_{index:03d}/motion_head_raw.npz'
        if p.exists() and q.exists():
            with np.load(p) as u,np.load(q) as v:
                keys=set(u.files)&set(v.files);different['motion_prediction']=any(u[k].shape!=v[k].shape or not np.allclose(u[k],v[k],rtol=0,atol=1e-4) for k in keys)
        for k,value in different.items():
            if value and first[k] is None:first[k]=index
        rows.append(dict(index=index,timestamp_s=index*.1,differences=different,
            ego_already_diverged=first['executed_state'] is not None and index>=first['executed_state'],
            interpretation='post-ego-divergence observations reflect both intervention and ego viewpoint' if first['executed_state'] is not None and index>=first['executed_state'] else 'matched actual ego state'))
    summaries={left:arm_summary(root/left),right:arm_summary(root/right)}
    prefix_end=int(round(read(root/'scenario/scenario.json')['config']['intervention_start_s']*10))
    prefix_files=list((root/right/'steps').glob('*/prefix_gate.json'))
    intervention_visible=first['camera_pixels'] is not None and (first['executed_state'] is None or first['camera_pixels']<first['executed_state'])
    front_entry=[]
    for index,r in bo.items():
        relative=r['environment'].get('target_relative_forward_left_up_m')
        target=r['environment'].get('target_world');geometry=b['vehicle_geometry']
        if relative and target:
            box=target['box'];pose=np.asarray(r['pose']);angle=box[6]-np.arctan2(pose[1,0],pose[0,0])
            half_forward=abs(np.cos(angle))*box[4]/2+abs(np.sin(angle))*box[3]/2
            half_lateral=abs(np.sin(angle))*box[4]/2+abs(np.cos(angle))*box[3]/2
            if relative[0]-half_forward>geometry['half_length_m']+geometry['clearance_margin_m'] and abs(relative[1])<geometry['half_width_m']+half_lateral:front_entry.append(index)
    actual_front_entry=bool(front_entry)
    if role=='external_patch':
        conclusion='patch_pair_anomaly_resolved_normal_A_check_still_required' if summaries[left]['any_anomaly'] and not summaries[right]['any_anomaly'] else 'patch_pair_requires_regression_and_recovery_review'
    elif summaries[left]['any_anomaly']:conclusion='baseline_anomaly_must_be_resolved_first'
    elif len(prefix_files)<prefix_end+1:conclusion='unresolved_common_history_or_prefix_protocol'
    elif not intervention_visible:conclusion='unresolved_intervention_not_verified_in_camera_observation'
    elif not actual_front_entry:
        conclusion='cutin_attempt_without_actual_front_entry_no_supported_failure_patch' if not summaries[right]['any_anomaly'] else 'unresolved_intervention_geometry_before_patch'
    elif summaries[right]['any_anomaly']:conclusion='intervention_anomaly_requires_repeat_and_mechanism_diagnosis'
    else:conclusion='no_supported_failure_mechanism_no_patch_generated; reasonable yielding may be correct'
    result=dict(completed=True,left=left,right=right,comparison_role=role,actor_tracks_identical=actor_tracks_identical,summaries=summaries,first_difference_step=first,
        first_difference_time_s={k:None if v is None else v*.1 for k,v in first.items()},
        intervention_verified_in_camera_observation=intervention_visible,scene_intervention_verification_applicable=role=='scene_intervention',common_prefix_gate_records=len(prefix_files),
        intervention_camera_difference_precedes_ego_divergence=intervention_visible,
        target_entered_actual_ego_front_corridor=actual_front_entry,first_actual_front_entry_step=min(front_entry) if front_entry else None,
        common_prefix_complete=len(prefix_files)>=prefix_end+1,conclusion=conclusion,
        candidate_patch_generated=(root/'candidate.json').is_file(),
        repeatable_mechanism_verified=(root/'mechanism.json').is_file() and read(root/'mechanism.json').get('status')=='repeatable_B_only_mechanism_supported',rows=rows)
    write(root/f'comparison_{left}_{right}.json',result);return result


def artifacts(root,left='A',right='B'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    import imageio_ffmpeg,subprocess
    root=Path(root);stage=root/f'presentation_{left}_{right}';stage.mkdir(exist_ok=True)
    if (stage/'artifacts.json').exists():
        from common import verify
        cached=read(stage/'artifacts.json')
        for a in [cached['video'],cached['overlay'],*cached['source_reports']]:verify(a)
        if not cached['full_decode_passed']:raise ValueError('Cached video was not fully decoded')
        return dict(video=cached['video']['path'],overlay=cached['overlay']['path'],reused=True)
    figure,axes=plt.subplots(figsize=(9,5))
    reports={name:read(root/name/'run_report.json') for name in [left,right]}
    for name,r in reports.items():
        p=np.array([o['pose'] for o in r['observations']]);axes.plot(p[:,0,3],p[:,1,3],label=name+' executed ego')
        with np.load(root/name/'actor_tracks.npz') as z:t=z['RigidNodes__instances_trans'][:len(p),3]
        axes.plot(t[:,0],t[:,1],'--',label=name+' target native3')
    axes.set(xlabel='Source-world forward x (m)',ylabel='Source-world left y (m)',title='Actual execution and native vehicle trajectories');axes.axis('equal');axes.legend();figure.tight_layout();figure.savefig(stage/'executed_trajectory_overlay.png',dpi=160);figure.savefig(stage/'executed_trajectory_overlay.svg');plt.close(figure)
    frames=stage/'frames';frames.mkdir(exist_ok=True);n=max(len(v['observations']) for v in reports.values())
    cameras=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']
    for index in range(n):
        sheet=Image.new('RGB',(1440,680),'#151515');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f't={index*.1:.1f}s | same simulation clock; actual ego viewpoints may diverge',fill='white')
        for row,name in enumerate([left,right]):
            draw.text((8,30+row*320),name+' actual closed-loop observations',fill='white')
            for col,camera in enumerate(cameras):
                path=root/name/f'steps/{index:03d}'/(camera+'.png')
                if path.exists():
                    with Image.open(path) as im:im.thumbnail((480,280));sheet.paste(im,(col*480,55+row*320))
                else:draw.text((col*480+10,150+row*320),'Arm terminated; no fabricated observation',fill='white')
        sheet.save(frames/f'{index:04d}.png')
    output=stage/'synchronized_3CAM.mp4'
    if output.exists():raise FileExistsError(output)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-framerate','10','-i',str(frames/'%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(output)],check=True,timeout=120)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','1','-i',str(output),'-f','null','-'],check=True,timeout=60)
    write(stage/'artifacts.json',dict(video=asset(output),overlay=asset(stage/'executed_trajectory_overlay.png'),full_decode_passed=True,
        source_reports=[asset(root/name/'run_report.json') for name in [left,right]]));return dict(video=str(output),overlay=str(stage/'executed_trajectory_overlay.png'))
