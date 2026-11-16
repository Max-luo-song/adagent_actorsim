"""GPU edit-sensor differential at equal ego poses/times; no UniAD or controls."""

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
import numpy as np
from platform_io import read,write,asset,verify
from optional_sensors import Recorder
from sensor_contracts import configuration

def validate(task,profile,work,event):
    from driver_adapter import stage_work
    provider=stage_work(task,profile,work)
    from native_car_edit import make_types
    from environment import install_trajectory,actor_arrays
    work=Path(work);out=work/'sensor_validation'
    if out.exists():raise FileExistsError('Existing sensor differential retained; use a new request ID')
    out.mkdir();c=configuration(task.get('sensors',{}).get('lidar'))
    if not c['enabled'] or len(task['edits'])!=1:raise ValueError('Validation requires enabled sensor and registered native edit')
    c['record_every_n_steps']=1
    context=read(work/'initial_context.json');pose=np.asarray(context['ego_to_world']);frames=[10,30,50]
    Backend,Session=make_types();session=Session(backend_factory=Backend);cleanup=dict(closed=False,reset_verified=False)
    samples={};actors={};status='failed'
    try:
        session.load_scene(work/'scene_manifest.json')
        for arm in ['original','edited','reset']:
            if arm!='original':session.reset()
            if arm=='edited':
                receipt=install_trajectory(session,work/'scenario/target_track.npz',1.5)
                write(out/'edit_receipt.json',receipt);event('native_vehicle_trajectory','edited',receipt)
            actors[arm]=actor_arrays(session)
            recorder=Recorder(c,out/arm)
            samples[arm]=[]
            for frame in frames:
                packet=session.render(pose,frame/session.spec.fps);record=recorder.record(session,packet)
                samples[arm].append(record);event('optional_pointcloud_recorder','captured',dict(arm=arm,frame=frame,points=record['point_count']))
        comparisons=[]
        for index,frame in enumerate(frames):
            arrays={}
            for arm in samples:
                with np.load(samples[arm][index]['points']['path']) as z:arrays[arm]={k:z[k].copy() for k in z.files if k not in ['source_timestamp_s','source_frame','local_timestamp_s']}
            def equal(left,right):return set(left)==set(right) and all(left[k].shape==right[k].shape and np.allclose(left[k],right[k],rtol=0,atol=1e-5) for k in left)
            comparisons.append(dict(source_frame=frame,original_equals_edited=equal(arrays['original'],arrays['edited']),
                original_equals_reset=equal(arrays['original'],arrays['reset']),edited_points=samples['edited'][index]['point_count']))
        unchanged=[]
        for key in actors['original']:
            a,b=actors['original'][key],actors['edited'][key]
            if key in ['RigidNodes__instances_trans','RigidNodes__instances_quats']:
                others=[i for i in range(a.shape[1]) if i!=3];unchanged.append(np.array_equal(a[:,others],b[:,others]))
            else:unchanged.append(np.array_equal(a,b))
        cleanup['reset_verified']=all(np.array_equal(actors['original'][k],actors['reset'][k]) for k in actors['original']) and all(r['original_equals_reset'] for r in comparisons)
        checks=dict(common_prefix_identical=comparisons[0]['original_equals_edited'],
            edited_returns_changed=any(not r['original_equals_edited'] for r in comparisons[1:]),
            points_nonempty=all(r['point_count']>0 for rows in samples.values() for r in rows),
            other_actors_unchanged=all(unchanged),reset_restores_points=cleanup['reset_verified'],
            shared_model_tensors=all(r['shared_actor_tensors'] and all(r['shared_actor_tensors'].values()) for rows in samples.values() for r in rows))
        result=dict(schema='optional_sensor_differential_v1',status='completed' if all(checks.values()) else 'failed',
            checks=checks,comparisons=comparisons,provider=c['provider'],control_executed=False,uniad_inference_executed=False,
            same_pose_and_same_source_time=True,physical_lidar_certified=False,feeds_controller=False,
            frame_receipts={arm:[asset(Path(r['points']['path']).parent/'receipt.json') for r in rows] for arm,rows in samples.items()})
        provider.check(work)
    finally:
        try:session.close();cleanup['closed']=True
        finally:write(out/'cleanup.json',cleanup)
    result['cleanup']=cleanup;write(out/'report.json',result)
    return dict(schema='sensor_validation_task_result_v1',task_id=task['task_id'],status=result['status'],
        execution=dict(control_executed=False,uniad_inference_executed=False,source_frames=frames),
        sensor_validation=result,recovery_required=not(cleanup['closed'] and cleanup['reset_verified']),
        artifacts=[dict(kind='sensor_edit_differential',**asset(out/'report.json'))])
