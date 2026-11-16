"""CPU external admission + full obstacle-memory replay on frozen observations.

Neural perception files are linked, never recomputed or rewritten. Independent
saved depth/unknown geometry is reused identically for both driver memories.
No new MPC, action execution, source actors or truth used by the policy.
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

import copy,datetime,importlib.util,json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;P28=HERE.parents[1]/'phase28';ROOT=P28.parents[5]
sys.path[:0]=[str(HERE.parent),str(P28),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from global_repair_v1.pipeline import EvidencePipeline
from global_repair_v1.driver import ContractObstacleMemory,validate_guard_input
from global_repair_v1.screening import retained_reference_check
from heading_fix_v1.trace_near_ego import guard
from heading_fix_v1 import closed172_common as cc
from planning_guard import ObstacleMemory
SOURCE=P28/'runs/proposal_evidence_inference_20261103T032120Z/baseline'
SENSORS=P28/'runs/lineage92_closed_scene552_v4'
SENSOR_FIELDS=('observed_depth_obstacles_xy','observed_depth_geometry','unknown_visibility_obstacles_xy','unknown_visibility_geometry')

def finite(value):return float(value) if np.isfinite(value) else None

def main():
    out=HERE.parent/'runs'/('global_evidence_replay_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    manifest=_hugsim_expand(json.loads((SENSORS/'scene_manifest.json').read_text()));source=_hugsim_expand(json.loads(Path(manifest['source_report']).read_text()))
    memories={mode:(ObstacleMemory if mode=='baseline' else ContractObstacleMemory).from_source(source) for mode in ('baseline','admission')}
    vehicle=cc.load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json');observer=None;frames=[];refs=changes=0
    spec=importlib.util.spec_from_file_location('frozen_velocity_gate',HERE/'velocity.py');velocity=importlib.util.module_from_spec(spec);spec.loader.exec_module(velocity)
    for i in range(81):
        d=SOURCE/f'step_{i:03d}'
        with (d/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        report=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text()));query=_hugsim_expand(json.loads((d/'neural_query_origin.json').read_text()))
        sensors=_hugsim_expand(json.loads((SENSORS/f'uniad_worker/step_{i:03d}/guard_memory_report.json').read_text()))
        with np.load(d/'perception.npz') as data:arrays={k:data[k].copy() for k in data.files}
        original_arrays={k:v.copy() for k,v in arrays.items()}
        if observer is None:observer=EvidencePipeline(report['config'])
        geometry=arrays.get('geometry_candidate_boxes',arrays['candidate_boxes'])
        raw=np.asarray(query['decoded_gravity_boxes'],dtype=np.float32).copy();raw[:,2]-=raw[:,5]/2
        filtered=observer.process(report,raw,geometry,arrays['candidate_scores'],arrays['candidate_labels'],query['selected_track_ids'],obs,info)
        snapshot=observer.capture();restored=EvidencePipeline(report['config']);restored.restore(snapshot)
        assert restored.capture()==snapshot
        if i==40:memories['admission']=pickle.loads(pickle.dumps(memories['admission'],protocol=4))
        check=retained_reference_check(report,filtered,info);refs+=check['reference_observations'];changes+=len(check['changed_or_removed'])
        frame=dict(step=i,quarantined=[q['neural_id'] for q in filtered['proposal_admission']['quarantined']],visible_reference_check=check,cases={})
        for mode,raw in [('baseline',report),('admission',filtered)]:
            target=out/mode/f'step_{i:03d}';target.mkdir(parents=True)
            (target/'perception.npz').symlink_to(d/'perception.npz');(target/'observation.pkl').symlink_to(d/'observation.pkl')
            (target/'scene_reliability.json').write_text(json.dumps(raw,indent=2))
            full=memories[mode].update(raw,geometry,arrays['candidate_scores'],arrays['candidate_labels'],info,current_occupancy=arrays['occupancy'].reshape(5,200,200)[0])
            if mode=='admission':validate_guard_input(full,info)
            for name in SENSOR_FIELDS:full[name]=copy.deepcopy(sensors[name])
            full=velocity.apply(full,info)
            (target/'guard_memory_report.json').write_text(json.dumps(full,indent=2))
            current,initial=guard(info,raw,arrays['occupancy'],vehicle);full_guard,_=guard(info,full,arrays['occupancy'],vehicle)
            obsgeom=full['observed_depth_geometry'];unkgeom=full['unknown_visibility_geometry']
            temporal=cc.TemporalOccupancy(arrays['occupancy'].reshape(1,5,200,200),vehicle,
                observed_cell_rotation=info['explicit_l2g_r_mat'] if obsgeom.get('policy')=='observed_depth_geometry_with_free_space_clearing_v1' and np.isclose(obsgeom.get('voxel_resolution_m',-1),.5) else None,
                observed_points=full['observed_depth_obstacles_xy'],observed_point_radius=obsgeom.get('voxel_half_diagonal_m',np.sqrt(2)*.25),
                unknown_cell_rotation=info['explicit_l2g_r_mat'] if unkgeom.get('policy')=='endpoint_supported_unknown_visibility_v2' and np.isclose(obsgeom.get('voxel_resolution_m',-1),.5) else None,
                unknown_points=full['unknown_visibility_obstacles_xy'],unknown_point_radius=unkgeom.get('point_radius_m',np.sqrt(2)*.25))
            raw_gap=float(temporal.clearance(initial[None],[0.])[0]);proposal_gap=float(full_guard.clearance(initial[None],[0.])[0])
            frame['cases'][mode]=dict(current_proposal_clearance_m=finite(current.clearance(initial[None],[0.])[0]),full_memory_proposal_clearance_m=finite(proposal_gap),occupancy_depth_unknown_clearance_m=finite(raw_gap),initial_constraint_clearance_m=finite(min(raw_gap,proposal_gap)),memory_tracks=len(memories[mode].tracks),sensor_constraints_reused_unchanged=True)
        for key in arrays:np.testing.assert_array_equal(arrays[key],original_arrays[key])
        frames.append(frame)
        if i%10==0:print('Admission + memory replay',i,flush=True)
    summary=dict(scope='CPU fixed-observation external admission replay; no new neural inference, MPC or actions',formal_evaluation_admitted=False,source=str(SOURCE),frames=len(frames),
        visible_retained_reference_observations=refs,visible_reference_changed_or_removed=changes,raw_arrays_unchanged=True,score_changes=0,
        global_repair_complete=False,closed_loop_executed=False,snapshot_roundtrips=81,driver_memory_pickle_roundtrip=True,external_obstacle_memory_replayed=True,independent_depth_unknown_geometry_reused=True,cases={})
    for mode in ('baseline','admission'):
        summary['cases'][mode]=dict(current_proposal_initial_conflict_frames=[r['step'] for r in frames if (r['cases'][mode]['current_proposal_clearance_m'] or 0)<-1e-5],
            full_memory_initial_conflict_frames=[r['step'] for r in frames if (r['cases'][mode]['initial_constraint_clearance_m'] or 0)<-1e-5])
    (out/'frames.json').write_text(json.dumps(frames,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True);print('OUTPUT:',out,flush=True)
if __name__=='__main__':main()
