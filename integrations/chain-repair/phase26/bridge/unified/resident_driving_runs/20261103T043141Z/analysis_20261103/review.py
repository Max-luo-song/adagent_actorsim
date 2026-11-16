"""CPU review of actual multi-episode resident driving; never reruns a model."""

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

import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle

import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=next(p for p in RUN.parents if (p/'code/HUGSIM').is_dir())
NAMES=['scene552_first','scene552_reset','scene551_switch','scene552_return']


def read(path):return _hugsim_expand(json.loads(path.read_text()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rgb(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))
def digest(array):return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    cache={};originals={}
    for path in RUN.rglob('*'):
        if not path.is_file() or OUT in path.parents or path.suffix not in ('.json','.png','.pkl','.npz'):continue
        stat=path.stat();key=(stat.st_dev,stat.st_ino)
        if key not in cache:cache[key]=sha(path)
        originals[str(path.relative_to(RUN))]=cache[key]
    top=read(RUN/'acceptance.json')
    ready=read(RUN/'inference_service/ready.json')
    service=read(RUN/'inference_service/inferences.json')
    leases=read(RUN/'inference_service/episodes.json')
    transitions=read(RUN/'scene_host/transitions.json')
    checks=dict(original_14_passed=len(top['checks'])==14 and top['passed'] and all(top['checks'].values()),
        twenty_service_inferences=len(service)==20 and [x['service_index'] for x in service]==list(range(20)),
        one_model_identity=ready['uniad_model_loads']==1 and all(x['identity']==ready['resident_identity'] for x in service),
        four_leases=len(leases)==4 and all(x['local_count']==5 and x['completed'] and x['worker_pid']==ready['pid'] for x in leases),
        one_scene_session=len(transitions)==4 and len({x['session_object_id'] for x in transitions})==1,
        same_backend_reset=transitions[1]['same_backend_reset'] and transitions[0]['backend_object_id']==transitions[1]['backend_object_id'],
        real_scene_sequence=[x['scene'] for x in transitions]==[552,552,551,552],
        loaded_models=bool(transitions[0]['gaussian_points_by_group']) and transitions[0]['gaussian_points_by_group']!=transitions[2]['gaussian_points_by_group'] and transitions[0]['gaussian_points_by_group']==transitions[3]['gaussian_points_by_group'])
    views=[];mappings=[];metrics={};source_checks={};receipts_all=[];reports=[]
    source_maps=[RUN/'implementation.json']
    for ep,name in enumerate(NAMES):
        folder=RUN/name;report=read(folder/'closed_loop/run_report.json');reports.append(report)
        acceptance=read(folder/'acceptance.json');weather=read(folder/'weather/report.json')['records']
        packets=read(folder/'weather_packet_audit.json');receipts=read(folder/'weather_input_receipts.json');receipts_all.append(receipts)
        reset=read(folder/'unified_reset_audit.json')
        source_maps.append(folder/'implementation.json')
        checks[name+'_acceptance']=acceptance['passed'] and all(acceptance['checks'].values())
        checks[name+'_feedback']=report['status']=='completed' and len(report['trace'])==5 and len(report['observations'])==6 and report['control_actions_executed'] and not report['recorded_future_ego_used_for_closed_motion']
        checks[name+'_driver_state']=len(report['driver_state_initial']['checks'])==10 and all(report['driver_state_initial']['checks'].values())
        checks[name+'_reset']=len(reset)==1 and all(reset[0].values())
        checks[name+'_pose']=len(weather)==6 and all(np.array_equal(w['ego_to_world'],o['closed_ego_to_world']) and w['timestamp']==o['timestamp'] for w,o in zip(weather,report['observations']))
        checks[name+'_causal']=len(weather)==6 and [w['audit']['window_frames'] for w in weather]==[[80,80,80],[80,80,81],[80,81,82],[81,82,83],[82,83,84],[83,84,85]]
        checks[name+'_source_stable']=report['implementation_unchanged'] and report['checkpoint_unchanged']
        initial=report['trace'][0]['state_before']
        checks[name+'_motion_start']=np.array_equal(initial[:3],[0,0,0]) and initial[3]==report['driver_state_initial']['initial_speed_mps'] and initial[4]==report['driver_state_initial']['initial_steering_rad']
        for i,w in enumerate(weather):
            for cam,c in w['cameras'].items():
                image=rgb(folder/f'weather/step_{i:03d}/{cam}_weather.png')
                clear=rgb(folder/f'weather/step_{i:03d}/{cam}_clear.png')
                with Image.open(folder/f'weather/step_{i:03d}/{cam}_mask.png') as im:mask=np.array(im)>0
                h=digest(image);receipt=receipts.get(f'step_{i:03d}')
                views.append(dict(episode=name,step=i,camera=cam,
                    harness=bool(np.array_equal(image,rgb(folder/f'closed_loop/closed_loop/{cam}/{i:03d}.png'))),
                    report_hash=h==c['rgb_sha256'],clear_hash=digest(clear)==c['clear_sha256'],
                    dynamic_pixels=bool(np.array_equal(image[mask],clear[mask])),
                    packet_hash=h==packets[i]['rgb_sha256'][cam],
                    receipt=None if receipt is None else h==receipt['rgb_sha256'][cam] and receipt['timestamp']==w['timestamp']))
        for local in range(5):
            global_index=ep*5+local;model=RUN/f'inference_service/worker/step_{global_index:03d}'
            step=folder/f'closed_loop/uniad_worker/step_{local:03d}'
            done=read(step.parent/f'done_{local:03d}.json');source=read(RUN/f'inference_service/worker/done_{global_index:03d}.json')
            state=read(step/'session_lifecycle.json');temporal=read(step/'temporal_state.json')
            mappings.append(dict(service_index=global_index,episode=name,local_index=local,
                done=done['status']=='ok' and done['index']==local and done['service_index']==global_index and source['status']=='ok',
                model_files=sha(step/'perception.npz')==sha(model/'perception.npz') and sha(step/'session_lifecycle.json')==sha(model/'session_lifecycle.json'),
                identity=state['identity']==ready['resident_identity'],
                history=temporal['history_bev_present']==(local>0) and state['reset']==(local==0),
                clean_reset=local>0 or state['reset_state_cleared'],
                lease=temporal['scene_token']==leases[ep]['token']==service[global_index]['episode_token'],
                reset_count=state['reset_count']==ep+1))
        with (folder/'closed_loop/uniad_worker/step_000/observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
        checks[name+'_first_packet']=info['scene_token']==leases[ep]['token'] and all(digest(im)==receipts['step_000']['rgb_sha256'][cam] for cam,im in obs['rgb'].items())
        positions=np.array([o['closed_ego_to_world'] for o in report['observations']])[:,:3,3]
        metrics[name]=dict(scene=report['scene'],controls=5,observations=6,simulation_seconds=.5,
            path_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
            initial_speed_mps=initial[3],final_speed_mps=report['trace'][-1]['state_after'][3],
            wall_seconds=report['wall_seconds'],minimum_actor_gap_m=report['minimum_all_actor_reference_gap_m'],
            minimum_predicted_road_clearance_m=min(t['predicted_road_swept_clearance_m'] for t in report['trace']))
    checks['all_72_views']=len(views)==72 and all(all(v for k,v in x.items() if k not in ('episode','step','camera','receipt')) for x in views)
    checks['all_60_input_receipts']=sum(x['receipt'] is True for x in views)==60
    checks['twenty_local_global_mappings']=len(mappings)==20 and all(all(v for k,v in x.items() if k not in ('service_index','episode','local_index')) for x in mappings)
    checks['same_scene_repeat_rgb']=receipts_all[0]==receipts_all[1]
    differences={key:max(float(np.max(np.abs(np.asarray(x[key])-np.asarray(y[key])))) for x,y in zip(reports[0]['trace'],reports[1]['trace'])) for key in ('state_before','state_after','neural_plan')}
    action_difference=max(abs(x['action'][key]-y['action'][key]) for x,y in zip(reports[0]['trace'],reports[1]['trace']) for key in x['action'])
    checks['same_scene_repeat_states_zero']=all(x==0 for x in differences.values()) and action_difference==0
    for source_map in source_maps:
        for path,h in read(source_map).items():
            source_checks[str(source_map.relative_to(RUN))+':'+path]=Path(path).is_file() and sha(Path(path))==h
    checks['source_fingerprints']=all(source_checks.values())
    checks['diagnostic_scope']=top['closed_loop_driving'] is True and not top['safety_certified'] and not top['formal_evaluation_admitted']
    checks['original_evidence_unchanged']=all(sha(RUN/path)==h for path,h in originals.items())
    result=dict(passed=all(checks.values()),checks=checks,metrics=metrics,views=views,mappings=mappings,
        repeat_differences=differences,repeat_action_max_difference=action_difference,
        source_checks=source_checks,original_evidence_sha256=originals,resident_pid=ready['pid'],
        input_evidence='60 submitted RGB receipts match; 12 RGB in four retained initial packets directly reread. Later packets were deleted normally by the harness.',
        closed_loop_driving=True,safety_certified=False,formal_evaluation_admitted=False)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result['passed'],check_count=len(checks),failed=[k for k,v in checks.items() if not v],metrics=metrics,repeat_differences=differences,repeat_action_max_difference=action_difference),indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
