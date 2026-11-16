"""Matched native score exports: counterfactual references and actual closed-loop pairs."""

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

import copy,pickle,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
from common import P26,BUNDLE,SCORER,asset,read,write,configure,sha
from environment import Evaluation

def export_pair(root,name,off,on=None,executed=False,replay=False):
    configure()
    import native_scores as native
    from scoring import native_plan_from_lidar
    from runtime_configuration import load_vehicle_config
    root=Path(root);off=Path(off);on=Path(on) if on is not None else None;vehicle=load_vehicle_config()
    left=read(off/'run_report.json');right=read(on/'run_report.json') if on is not None else left
    target=root/'evaluations'/name;target.mkdir(parents=True,exist_ok=True)
    n=min(len(left['trace']),len(right['trace']),164)
    if executed:n=min(n,len(left['observations'])-25,len(right['observations'])-25)
    if n<=0:raise ValueError('No common valid score origins')
    with np.load(off/'actor_tracks.npz') as z:tracks={k:z[k].copy() for k in z.files}
    if on is not None:
        with np.load(on/'actor_tracks.npz') as z:
            if set(z.files)!=set(tracks) or any(not np.array_equal(z[k],tracks[k]) for k in z.files):raise ValueError('Patch pair changed actor state')
    geometry=Evaluation.__new__(Evaluation);geometry.tracks=tracks;geometry.vehicle=vehicle
    route=np.load(BUNDLE/'native_dense_route.npy')[:,[0,2],3]
    def rc(pose):
        xy=np.asarray(pose)[:2,3];index=int(np.linalg.norm(route-[-xy[1],xy[0]],axis=1).argmin());return (index+1)/(.9*len(route))
    streams={};links=[]
    for kind,folder,report in [('raw_uniad',off,left),('adopted_plan',on or off,right)]:
        observations={r['index']:r for r in report['observations']};keyframes={}
        for i in range(n):
            with (folder/f'uniad_worker/step_{i:03d}/sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
            reference=read(folder/f'steps/{i:03d}/reference.json')['adopted']
            if replay and kind=='adopted_plan':reference=read(root/f'replay/{name}/references/{i:03d}.json')['patched_reference']
            frame=native_plan_from_lidar(native,reference,info,vehicle)
            if executed:
                values=[]
                for delta in [5,10,15,20,25]:
                    pose=np.asarray(observations[i+delta]['pose']);yaw=float(np.arctan2(pose[1,0],pose[0,0]));values.append([float(pose[0,3]),float(pose[1,3]),yaw])
                frame['planned_traj']=dict(traj=values,timestep=.5);frame['trajectory_source']='actual_recorded_Motion_execution'
            frame['plan_kind']=('patch_off' if kind=='raw_uniad' else 'patch_on')+('_actual_execution' if executed else '_reference')
            keyframes[i]=frame
        final=n-1+25;frames=[];last_actual=max(observations);final_rc=max(rc(o['pose']) for o in report['observations'])
        for i in range(final+1):
            if i in keyframes:frame=copy.deepcopy(keyframes[i])
            else:
                o=observations.get(i,observations[last_actual]);pose=np.asarray(o['pose']);yaw=float(np.arctan2(pose[1,0],pose[0,0]))
                frame=dict(time_stamp=i*.1,is_key_frame=False,ego_box=[*pose[:3,3].tolist(),2*vehicle.half_width_m,2*vehicle.half_length_m,vehicle.body_height_m,yaw],planned_traj=dict(traj=[],timestep=.5))
            boxes=geometry.boxes(i+10);frame.update(obj_boxes=[b['box'] for b in boxes],
                obj_names=['car' if b['group']=='RigidNodes' else 'pedestrian' for b in boxes],rc=rc(observations[i]['pose']) if i in observations else final_rc)
            frames.append(frame)
        streams[kind]=[dict(frames=frames,metas=dict(actual_execution=executed,shared_current_observations=replay,
            role='off' if kind=='raw_uniad' else 'on',reference_counterfactual_only=replay and kind=='adopted_plan'))]
    manifest=copy.deepcopy(read(off/'native_eval/export_manifest.json'))
    manifest.update(pairs=n,confirmed_controls=0 if replay else min(len(left['trace']),len(right['trace'])),
        links=[dict(off=asset(off/'run_report.json'),on=asset(on/'run_report.json') if on is not None else asset(root/f'replay/{name}/report.json'))],
        episode_complete=not replay and left['status']=='completed' and right['status']=='completed',
        actual_execution_compared=executed,offline_fixed_observation_reference_pair=replay,
        patch_future_actor_truth_used_online=False,source_frame_offset=10,score_origin_count=n,
        score_origin_range=[0,n-1],scoring_scope='unchanged native metrics; same source clocks/actor programs; hypothetical references never counted as executed ego poses')
    # The native API requires a nonempty trace gate. These rows explicitly reference
    # existing observations; they contain no newly executed controls or fake ego states.
    write(target/'run_report.json',dict(schema='native_pair_score_request_only',real_closed_loop=False,
        actual_controls_executed_by_this_request=0,source_actual_reports=[asset(off/'run_report.json')]+([asset(on/'run_report.json')] if on else []),
        trace=[dict(observation_index=i,role='score_request_not_a_new_control') for i in range(n)]))
    write(target/'export_manifest.json',manifest)
    for kind,data in streams.items():
        with (target/(kind+'_data.pkl')).open('wb') as f:pickle.dump(data,f,protocol=4)
    return target,streams,manifest,vehicle

def score_pair(root,name,off,on=None,executed=False,replay=False):
    target,data,manifest,vehicle=export_pair(root,name,off,on,executed,replay)
    (target/'native_eval').mkdir(exist_ok=True)
    import native_scores as native
    export=native.export;began=time.monotonic()
    try:
        native.export=lambda *args:(data,manifest)
        result=native.score(target,vehicle)
    finally:native.export=export
    result.update(score_origin_count=manifest['score_origin_count'],off_on_labels=dict(raw_uniad='patch_off',adopted_plan='patch_on'),
        actual_execution_compared=executed,offline_fixed_observation_pair=replay,
        paired_same_observations=replay,score_request_executed_new_controls=False,
        evaluation_seconds=time.monotonic()-began,manifest=asset(target/'export_manifest.json'))
    write(target/'pair_scores.json',result);return result
