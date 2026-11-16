"""Replay saved sensor packets to expose and validate the existing map head.

No edits, simulator object coordinates, or recorded future ego enter control.
Reference geometry below is used only to evaluate the selected candidate.
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

pass  # Path is provided by the portable boundary above
import os,sys,json,pickle,subprocess,time,hashlib
import numpy as np
from scene_api import BASE
A=BASE/'waymo23-native-import'
sys.path[:0]=[str(A),str(A/'waymo172_closed_loop_v1')]
from closed172_common import atomic_json,atomic_pickle,wait_json,optimize,initial_native_pose,native_delta,reference_check
from drivable_area import from_saved,DrivableArea
from runtime_configuration import load_vehicle_config
from native_geometry_evaluation import NativeVehicleEvaluation
from waymo_adapter import SR_TO_HUGSIM as T
from uniad_bridge import placement_reference


def run(request_path,output):
    req=_hugsim_expand(json.loads(Path(request_path).read_text()));output=Path(output).resolve()
    if set(req)!={'schema','source_result','native_geometry','actor_geometry'} or req['schema']!='uniad_road_constraint_preflight_v1':raise ValueError('Invalid road preflight request')
    source=Path(req['source_result']).resolve();geometry=Path(req['native_geometry']).resolve();actor_geometry=Path(req['actor_geometry']).resolve()
    for p in [source,geometry,actor_geometry]:
        if BASE not in p.parents:raise ValueError('Preflight inputs must be workspace artifacts')
    report=_hugsim_expand(json.loads((source/'run_report.json').read_text()))
    if report.get('start_frame')!=80 or report.get('time_policy')!='source_time_only':raise ValueError('This diagnostic replays the declared v20 dynamic case')
    output.mkdir(parents=True,exist_ok=False);job=output/'uniad_worker';job.mkdir();(job/'step_000').mkdir()
    v=load_vehicle_config();process=None;stream=None;rows=[];first_info=None;first_obs=None
    status=dict(schema='road_constraint_preflight_result_v1',status='running',source_result=str(source),control_executed=False,ground_truth_used_for_control=False)
    atomic_json(output/'run_report.json',status)
    try:
        with (source/'uniad_worker/step_000/observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        first_info=info;first_obs=obs
        atomic_pickle(job/'step_000/observation.pkl',(obs,info))
        env=dict(os.environ,BASE=str(BASE),PYTHONUNBUFFERED='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1',UNIAD_STARTUP_REPORT=str(job/'startup.json'))
        for key in ['LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']:env.pop(key,None)
        stream=(output/'uniad_worker.log').open('w')
        process=subprocess.Popen([str(BASE/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),str(A/'waymo172_closed_loop_v1/closed172_worker.py'),'--job',str(job),'--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],env=env,cwd=BASE,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(output/'progress.json',dict(stage='loading_uniad_map_preflight'))
        wait_json(job/'ready.json',process,timeout=1200.)
        from PIL import Image
        for index in range(3):
            folder=job/f'step_{index:03d}';folder.mkdir(exist_ok=True)
            if index:
                original=source/'uniad_worker'/f'step_{index:03d}'
                with (original/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
                obs={}
                for field,name in [('depth','observed_depth'),('depth_alpha','observed_depth_alpha'),('depth_variance','observed_depth_variance'),('surface_depth','surface_depth'),('surface_valid','surface_valid')]:
                    path=original/(name+'.npz')
                    if path.exists():
                        with np.load(path) as arrays:obs[field]={k:arrays[k] for k in arrays.files}
                obs['rgb']={camera:np.array(Image.open(source/'closed_loop'/camera/f'{index:03d}.png').convert('RGB')) for camera in ('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT')}
                atomic_pickle(folder/'observation.pkl',(obs,info))
            request=dict(index=index,packet=str(folder/'observation.pkl'))
            if index==0:request['diagnostic_seed']=20261021
            atomic_json(job/f'request_{index:03d}.json',request)
            atomic_json(output/'progress.json',dict(stage='infer_predicted_road',step=index))
            done=wait_json(job/f'done_{index:03d}.json',process,timeout=300.)
            with np.load(done['perception']) as arrays:
                road=DrivableArea(from_saved(arrays),v);offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];initial=np.r_[-offset,0.,info['ego_velo'],info['ego_steer']]
                clearance=float(road.clearance(initial[None])[0]);audit=road.audit()
                rows.append(dict(step=index,timestamp=float(info['timestamp']),initial_road_clearance_m=clearance,audit=audit))
            print('Predicted road',index,'initial footprint clearance',clearance,flush=True)
        atomic_json(job/'stop.json',{})
        process.wait(timeout=30.)
        with np.load(job/'step_000/perception.npz') as first_arrays:
            first_data={k:first_arrays[k] for k in first_arrays.files}
        from road_preflight_plot import render
        render(output,source,first_data,from_saved(first_data),dict(selected=None))
        if any(row['initial_road_clearance_m']< -1e-5 or row['audit']['road_cells']==0 or row['audit']['non_road_cells']==0 for row in rows):
            status.update(status='failed',reason='road_prediction_or_alignment_not_admissible',road_rows=rows)
            atomic_json(output/'run_report.json',status)
            raise RuntimeError(status['reason'])
        with np.load(job/'step_000/perception.npz') as arrays:
            data={k:arrays[k] for k in arrays.files}
        road_contract=from_saved(data);road=DrivableArea(road_contract,v)
        with np.load(source/'uniad_worker/step_000/perception.npz') as old_arrays:
            same_occupancy=np.array_equal(data['occupancy'],old_arrays['occupancy'])
            plan_diff=float(np.max(abs(data['trajectory']-old_arrays['trajectory'])))
            same_candidates=True
            for key in ('geometry_candidate_boxes','candidate_scores','candidate_labels'):
                if (key in data)!=(key in old_arrays):same_candidates=False;continue
                if key in data:
                    same_candidates&=data[key].shape==old_arrays[key].shape and np.allclose(data[key],old_arrays[key],rtol=0.,atol=1e-4)
        status['replay_check']=dict(neural_plan_max_abs_diff_m=plan_diff,occupancy_identical=bool(same_occupancy),candidate_geometry_unchanged=bool(same_candidates))
        if plan_diff>1e-4 or not same_occupancy or not same_candidates:
            status.update(reason='archived_perception_replay_changed')
            raise RuntimeError('Archived perception changed; do not mix old reliability with new inference')

        reliability=_hugsim_expand(json.loads((source/'uniad_worker/step_000/guard_memory_report.json').read_text()))
        reliability.update(require_predicted_drivable_area=True,predicted_drivable_area=road_contract)
        atomic_json(output/'candidate_reliability.json',reliability)
        atomic_json(output/'progress.json',dict(stage='optimizing_road_constrained_candidate'))
        candidate=optimize(data['trajectory'],data['occupancy'],first_info,reliability=reliability,vehicle=v)
        atomic_json(output/'initial_road_optimization.json',candidate)
        selected=candidate['selected'];checks=dict(initial_road_supported=True,all_three_map_frames_admissible=True,candidate_feasible=selected is not None)
        if selected is not None:
            states=np.array(selected['states']);times=np.arange(len(states))*v.dt
            checks.update(road_footprint_clearance_m=float(road.clearance(states).min()),progress_m=float(states[-1,1]-states[0,1]),left_lateral_extent_m=float(states[:,0].min()-states[0,0]))
            # Independent references evaluate this fixture only; they never
            # change the seed, objective, obstacle memory or chosen controls.
            native=NativeVehicleEvaluation(dict(np.load(geometry)))
            asset=_hugsim_expand(json.loads((source/'asset_reference.json').read_text()));initial=initial_native_pose(first_info);mount=np.asarray(first_info['model_lidar_frame']['virtual_to_ego'])
            from native_actor_evaluation import NativeActorEvaluation
            actors=NativeActorEvaluation.from_archive(dict(np.load(actor_geometry)))
            native_gaps=[];asset_gaps=[];actor_gaps=[];overlap=False
            for i,state in enumerate(states):
                pose=initial@native_delta(state[0]+mount[0,3],state[1]+mount[1,3],state[2]);nc=native.check(T.T@pose@T,80+i,v);ac=reference_check(pose,asset,v)
                if nc['nearest']['gap_m'] is not None:native_gaps.append(nc['nearest']['gap_m'])
                actor=actors.check(T.T@pose@T,80+i,v)
                if actor['nearest']['gap_m'] is not None:actor_gaps.append(actor['nearest']['gap_m'])
                asset_gaps.append(ac['gap_m']);overlap|=nc['overlap'] or ac['overlap'] or actor['overlap']
            checks.update(posthoc_native_gap_m=min(native_gaps),posthoc_inserted_gap_m=min(asset_gaps),posthoc_overlap=bool(overlap),posthoc_all_actor_gap_m=min(actor_gaps) if actor_gaps else None)
        # Fixture-level gate, not a scene-specific planner steering rule.
        admitted=selected is not None and checks['road_footprint_clearance_m']>=-1e-5 and checks['progress_m']>=2*v.half_length_m and checks['left_lateral_extent_m']<=-.5 and not checks['posthoc_overlap'] and checks['posthoc_native_gap_m']>=v.clearance_margin_m and checks['posthoc_inserted_gap_m']>=v.clearance_margin_m and checks['posthoc_all_actor_gap_m'] is not None and checks['posthoc_all_actor_gap_m']>=v.clearance_margin_m
        with np.load(source/'uniad_worker/step_000/perception.npz') as old:
            checks['first_frame_neural_plan_max_abs_diff']=float(np.max(abs(data['trajectory']-old['trajectory'])))
        old_states=np.array([report['trace'][0]['state_before']]+[row['state_after'] for row in report['trace']]);offset=np.asarray(first_info['model_lidar_frame']['virtual_to_ego'])[:2,3];old_states[:,:2]-=offset
        checks['archived_driven_trajectory_initial_map_clearance_m']=float(road.clearance(old_states).min())
        status.update(status='passed' if admitted else 'failed',reason='ready_for_dynamic_feedback' if admitted else 'no_admissible_progressing_left_candidate',road_rows=rows,checks=checks)
        atomic_json(output/'run_report.json',status)
        atomic_json(output/'progress.json',dict(stage='completed',admitted=bool(admitted)))
        from road_preflight_plot import render
        render(output,source,data,road_contract,candidate)
        if not admitted:raise RuntimeError(status['reason'])
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));atomic_json(output/'run_report.json',status);raise
    finally:
        atomic_json(job/'stop.json',{})
        if process is not None and process.poll() is None:
            try:process.wait(timeout=20.)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=10.)
        if stream is not None:stream.close()
