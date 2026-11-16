"""Reuse the frozen native evaluator, explicitly distinguish plans and executed paths."""

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

import copy,hashlib,pickle,time,signal
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import numpy as np
from common import *
from environment import Evaluation


def score_arm(root,name):
    configure();root=Path(root);out=root/name;report=read(out/'run_report.json');n=len(report['trace'])
    if not n:
        result=dict(status='not_scored_no_executed_controls',scores=None,actual_executed_path_score=None)
        write(out/'native_eval/native_scores.json',result);return result
    from scene_api import SceneSpec
    import native_scores as native
    if Path(native.__file__).resolve()!=SOURCE/'native_scores.py':raise ValueError('Wrong native-score implementation')
    vehicle=SceneSpec.read(root/'scene_manifest.json').vehicle
    with np.load(out/'actor_tracks.npz') as z:tracks={k:z[k].copy() for k in z.files}
    geometry=Evaluation.__new__(Evaluation);geometry.tracks=tracks;geometry.vehicle=vehicle
    observations={r['index']:r for r in report['observations']};states={}
    for index,r in observations.items():
        path=out/f'uniad_worker/step_{index:03d}/sensor_metadata.pkl'
        if path.exists():
            with path.open('rb') as f:info=pickle.load(f)
            states[index]=native.raw_plan(np.zeros((6,2)),info,vehicle)['ego_box']
        else:
            pose=np.asarray(r['pose']);states[index]=[*pose[:3,3],2*vehicle.half_width_m,2*vehicle.half_length_m,vehicle.body_height_m,float(np.arctan2(pose[1,0],pose[0,0]))]
    last_actual=max(states);can_execute_score=last_actual>=25
    final_key=last_actual-25 if can_execute_score else n-1
    keyframes={kind:{} for kind in ['raw_uniad','adopted_plan']+(['executed_path'] if can_execute_score else [])};links=[]
    for i in range(final_key+1):
        folder=out/f'uniad_worker/step_{i:03d}'
        with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
        reference=read(out/f'steps/{i:03d}/reference.json')
        for kind,field in [('raw_uniad','raw_uniad'),('adopted_plan','adopted')]:
            frame=native.raw_plan(reference[field],info,vehicle);frame['plan_kind']=kind;keyframes[kind][i]=frame
        if can_execute_score:
            frame=copy.deepcopy(keyframes['raw_uniad'][i]);frame['plan_kind']='actual_executed_path'
            frame['planned_traj']=dict(traj=[[states[i+j][0],states[i+j][1],states[i+j][6]] for j in [5,10,15,20,25]],timestep=.5)
            for k in ['source_nn_nodes','source_nn_horizon_s']:frame.pop(k,None)
            frame['trajectory_source']='actual_recorded_Motion_states'
            keyframes['executed_path'][i]=frame
        links.append(dict(index=i,metadata=asset(folder/'sensor_metadata.pkl'),perception=asset(folder/'perception.npz'),reference=asset(out/f'steps/{i:03d}/reference.json')))
    last=final_key+25;data={};final_rc=max(native.route_completion(v[:2]) for v in states.values())
    for kind,keys in keyframes.items():
        frames=[]
        for i in range(last+1):
            if i in keys:value=copy.deepcopy(keys[i])
            else:value=dict(time_stamp=i/10.,is_key_frame=False,ego_box=states.get(i,states[last_actual]),planned_traj=dict(traj=[],timestep=.5))
            boxes=geometry.boxes(i+int(read(root/'plan.json')['initial_context']['source_frame']));value.update(obj_boxes=[r['box'] for r in boxes],obj_names=['car' if r['group']=='RigidNodes' else 'pedestrian' for r in boxes],
                rc=native.route_completion(states[i][:2]) if i in states else final_rc)
            frames.append(value)
        data[kind]=[dict(frames=frames,metas=dict(plan_kind=kind,shared_current_observations=True,
            actual_execution=kind=='executed_path',future_ego_values_only_from_actual_recording=True))]
    plan=read(root/'plan.json');weights=plan['assets']['uniad_weights']
    manifest=dict(native_source_sha256=sha(SCORER),scenario_sha256=sha(root/'scenario/scenario.json'),source_scene_manifest_sha256=sha(BUNDLE/'source_manifest.json'),
        ego_geometry_sha256=hashlib.sha256(str([vehicle.half_width_m,vehicle.half_length_m,vehicle.body_height_m]).encode()).hexdigest(),
        uniad_seed=plan['seed'],uniad_weights_sha256=weights['sha256'],static_background_sha256=sha(BUNDLE/'background_sr_world.npy'),
        static_road_sha256=sha(BUNDLE/'road_sr_world.npy'),point_semantics_sha256=sha(BUNDLE/'point_semantics/background_semantics.npz'),
        actor_values_sha256=sha(out/'actor_tracks.npz'),plan_horizon_s=2.5,native_export_helper_sha256=sha(SCORER.parent/'sim_utils.py'),
        episode_complete=report['status']=='completed',confirmed_controls=n,pairs=len(links),links=links,
        executed_path_scored=can_execute_score,executed_path_keyframes=final_key+1 if can_execute_score else 0,
        physical_road_boundary_verified=False,source_frame_offset=read(root/'plan.json')['initial_context']['source_frame'],scoring_scope='unchanged native metrics on reconstruction; raw/adopted are plans, executed_path uses only actual future execution')
    folder=out/'native_eval';folder.mkdir(exist_ok=True)
    write(folder/'export_manifest.json',manifest)
    for kind,value in data.items():
        with (folder/(kind+'_data.pkl')).open('wb') as f:pickle.dump(value,f,protocol=4)
    original=native.export
    began=time.monotonic()
    from runtime import TERMINATION
    previous_handler=signal.getsignal(signal.SIGALRM)
    def expired(*args):raise TimeoutError('Native scoring exceeded the preregistered wall resource limit')
    try:
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,TERMINATION['native_score_wall_limit_s'])
        native.export=lambda *args:(data,manifest)
        result=native.score(out,vehicle)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler);native.export=original
    result.update(executed_path_scored=can_execute_score,executed_path_score_scope='actual executed future .5s nodes, only where full 2.5s execution is recorded' if can_execute_score else 'insufficient executed horizon; no fabricated future execution',
        raw_and_adopted_scores_are_planning_scores=True,matched_keyframe_count=final_key+1,
        actual_controls=n,actual_progress=final_rc,physical_road_boundary_verified=False,
        geometry_semantics='reconstruction and predicted semantic bounds; not independently surveyed HD boundaries',native_scoring_wall_seconds=time.monotonic()-began)
    write(folder/'native_scores.json',result);return result
