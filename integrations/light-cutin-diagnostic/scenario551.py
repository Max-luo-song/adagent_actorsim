"""C2 native-car scripted trajectory intervention; no actor insertion."""

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

import numpy as np
import yaml
pass  # Path is provided by the portable boundary above
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from shapely.geometry import Polygon
from common import BUNDLE,read,write,asset

CONFIG=dict(scene=551,target='native:RigidNodes:3',model_index=3,source_instance_index=5,
    annotation_id='o24FGKGBO7aPlRVYkDyU7A',fps=10.,initial_ego_speed_mps=16.,
    intervention_start_s=1.2,lane_change_duration_s=2.8,actor_deceleration_start_s=3.8,actor_deceleration_duration_s=3.5,
    target_lane_center_source_y_m=.3,nominal_final_actor_speed_mps=10.,
    direction='source negative-y: from ego left lane toward ego lane; model right-positive',
    interpretation='scripted native-vehicle trajectory intervention, not a reactive traffic policy',
    observation_end_s=7.,future_score_support_s=2.5,
    actor_motion_limits=dict(max_speed_mps=35.,max_acceleration_mps2=8.,max_yaw_rate_radps=.8),
    no_new_actor=True,road_weather_others_unchanged=True)

def smooth(u):return 10*u**3-15*u**4+6*u**5
def integral(u):return 2.5*u**4-3*u**5+u**6

def polygon(t,q,size):
    l,w,h=size;corners=np.array([[a*l/2,b*w/2,0.] for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    return Polygon((Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).apply(corners)+t)[:,:2])

def generate(output):
    output=__import__('pathlib').Path(output);output.mkdir(parents=True,exist_ok=True)
    source=read(BUNDLE/'source_manifest.json');identity=read(source['instances_info.json']['path'])['5']
    if source['scene']!=551 or identity['id']!=CONFIG['annotation_id'] or identity['class_name']!='Vehicle':raise ValueError('Scene551 native identity mismatch')
    p0=np.loadtxt(Path(source['raw_data'])/'ego_pose/000.txt');p1=np.loadtxt(Path(source['raw_data'])/'ego_pose/001.txt')
    recorded_speed=float(np.linalg.norm((np.linalg.inv(p0)@p1)[:2,3])*CONFIG['fps'])
    with np.load(BUNDLE/'actors.npz') as z:arrays={k:z[k].copy() for k in z.files}
    t=arrays['RigidNodes__instances_trans'];q=arrays['RigidNodes__instances_quats'];visible=arrays['RigidNodes__instances_fv'];size=arrays['RigidNodes__instances_size'];index=CONFIG['model_index']
    if not visible[:,index].all() or len(t)!=199:raise ValueError('Expected full visibility on this original car')
    annotation_pose=np.linalg.inv(p0)@np.asarray(identity['frame_annotations']['obj_to_world'][0])
    np.testing.assert_allclose(annotation_pose[:3,3],t[0,index],rtol=0,atol=1e-5)
    np.testing.assert_allclose(identity['frame_annotations']['box_size'][0],size[index],rtol=0,atol=1e-5)
    times=np.arange(len(t))/CONFIG['fps']
    # Raw untrained heldout roots are not the poses rendered by this checkpoint.
    # Use its real source split/interpolation for continuous-path auditing/design.
    config_path=Path(source['source_checkpoint']).parent/'config.yaml';cfg=yaml.safe_load(config_path.read_text())
    stride=int(cfg['data']['pixel_source']['test_image_stride']);tests=list(range(stride,len(t),stride)) if stride else []
    effective_t=t[:,index].copy();effective_q=q[:,index].copy()
    for f in tests:
        if 1<f<len(t)-1 and visible[f-1,index] and visible[f+1,index]:
            effective_t[f]=(t[f-1,index]+t[f+1,index])/2
            effective_q[f]=Slerp([0,1],Rotation.from_quat(q[[f-1,f+1],index][:,[1,2,3,0]]))([.5]).as_quat()[0][[3,0,1,2]]
    curve=CubicSpline(times,effective_t,axis=0);rot=Slerp(times,Rotation.from_quat(effective_q[:,[1,2,3,0]]))
    start=CONFIG['intervention_start_s'];brake_start=CONFIG['actor_deceleration_start_s'];duration=CONFIG['actor_deceleration_duration_s']
    v0=float(np.linalg.norm(curve(brake_start,1)[:2]));ratio=CONFIG['nominal_final_actor_speed_mps']/v0
    def evaluate(clock):
        clock=np.asarray(clock);u=np.clip((clock-brake_start)/duration,0,1)
        source_time=clock+(ratio-1)*duration*integral(u)+(ratio-1)*np.maximum(clock-brake_start-duration,0)
        base=curve(source_time);ul=np.clip((clock-start)/CONFIG['lane_change_duration_s'],0,1)
        value=base.copy();value[:,1]+=(CONFIG['target_lane_center_source_y_m']-base[:,1])*smooth(ul)
        return value,source_time
    positions,retime=evaluate(times);dense=np.arange(0,CONFIG['observation_end_s']+CONFIG['future_score_support_s']+.005,.01)
    dense_pos,_=evaluate(dense);velocity=np.gradient(dense_pos,.01,axis=0);acceleration=np.gradient(velocity,.01,axis=0)
    yaw=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]));rate=np.gradient(yaw,.01)
    base_velocity=curve(retime,1);actual_velocity=np.gradient(positions,.1,axis=0)
    delta=np.unwrap(np.arctan2(actual_velocity[:,1],actual_velocity[:,0])-np.arctan2(base_velocity[:,1],base_velocity[:,0]))
    rotations=Rotation.from_rotvec(np.c_[np.zeros(len(times)),np.zeros(len(times)),delta])*rot(retime)
    quats=rotations.as_quat()[:,[3,0,1,2]]
    prefix=times<=start+1e-10;positions[prefix]=t[prefix,index];quats[prefix]=q[prefix,index]
    rendered_quats=quats.copy()
    for f in tests:
        if times[f]<start:rendered_quats[f]=effective_q[f]
    end=int(round(dense[-1]*10))+1
    actual_yaw=np.unwrap(Rotation.from_quat(rendered_quats[:end][:,[1,2,3,0]]).as_euler('xyz')[:,2])
    actual_body_yawrate=float(np.abs(np.diff(actual_yaw)/.1).max())
    # Collision audit is against OTHER reconstructed actors, not against an assumed ego path.
    conflicts=[];minimum=float('inf')
    for f in range(int(round(dense[-1]*10))+1):
        a=polygon(positions[f],quats[f],size[index])
        for other in np.flatnonzero(visible[f]):
            if other==index:continue
            if abs(t[f,other,2]-positions[f,2])>(size[other,2]+size[index,2])/2:continue
            b=polygon(t[f,other],q[f,other],size[other]);distance=float(a.distance(b));minimum=min(minimum,distance)
            if a.intersects(b):conflicts.append(dict(frame=f,other_model_index=int(other)))
    limits=CONFIG['actor_motion_limits'];metrics=dict(max_speed_mps=float(np.linalg.norm(velocity,axis=1).max()),
        max_acceleration_mps2=float(np.linalg.norm(acceleration,axis=1).max()),max_yaw_rate_radps=float(np.abs(rate).max()),max_body_yaw_rate_radps=actual_body_yawrate)
    checks=dict(native_identity_verified=True,prefix_poses_bitwise_unchanged=np.array_equal(positions[prefix],t[prefix,index]) and np.array_equal(quats[prefix],q[prefix,index]),
        speed_within_budget=metrics['max_speed_mps']<=limits['max_speed_mps'],acceleration_within_budget=metrics['max_acceleration_mps2']<=limits['max_acceleration_mps2'],
        yaw_rate_within_budget=max(metrics['max_yaw_rate_radps'],actual_body_yawrate)<=limits['max_yaw_rate_radps'],no_other_native_rigid_collision=not conflicts)
    np.savez_compressed(output/'target_track.npz',original_trans=t[:,index],original_quats=q[:,index],edited_trans=positions,edited_quats=quats,times=times,original_effective_trans=effective_t,original_effective_quats=effective_q)
    report=dict(config=CONFIG,checks=checks,passed=all(checks.values()),metrics=metrics,
        source_scene=asset(BUNDLE/'source_manifest.json'),source_actors=asset(BUNDLE/'actors.npz'),
        source_identity=source['instances_info.json'],initial_target_source_position_m=t[0,index].tolist(),
        annotation_to_model_initial_center_error_m=float(np.linalg.norm(annotation_pose[:3,3]-t[0,index])),
        annotation_to_model_dimensions_verified=True,
        source_config=asset(config_path),source_test_stride=stride,heldout_pose_interpolation_reproduced=True,
        nominal_initial_longitudinal_center_gap_m=float(t[0,index,0]),initial_left_separation_m=float(t[0,index,1]),
        onset_nominal_center_gap_m=float(curve(start)[0]-CONFIG['initial_ego_speed_mps']*start),
        nominal_actor_speed_at_onset_mps=float(np.linalg.norm(curve(start,1)[:2])),nominal_actor_speed_at_deceleration_onset_mps=v0,
        source_time_retiming=True,other_actor_conflicts=conflicts,
        recorded_source_initial_speed_mps=recorded_speed,controlled_ego_initial_speed_mps=CONFIG['initial_ego_speed_mps'],
        initial_speed_design='same 16m/s in both arms, below source recorded speed, allowing the original left-rear car to pass ahead before merging',
        minimum_other_rigid_polygon_distance_m=minimum if np.isfinite(minimum) else None,
        audit_end_time_s=float(dense[-1]),ego_gap_is_nominal_only_until_real_closed_loop=True,
        actor_height_source='retimed original reconstructed road profile; new-lane height not independently certified',
        physical_road_boundary_verified=False,gpu_render_sync_verified=False)
    write(output/'scenario.json',report)
    if not report['passed']:raise ValueError('Native-car intervention audit failed; inspect scenario.json')
    return report
