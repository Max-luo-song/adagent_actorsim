"""Offline-only native551 cut-in design. Never imported by the online controller."""

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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from shapely.geometry import Polygon
from common import BUNDLE, read, write, asset
from action_program import Timeline,validate_phases

CONFIG = dict(scene=551, target='native:RigidNodes:3', model_index=3,
    source_instance_index=5, annotation_id='o24FGKGBO7aPlRVYkDyU7A', fps=10.,
    initial_ego_speed_mps=4., initial_source_frame=10, intervention_start_s=.5,
    absolute_source_activation_s=1.5, lane_change_duration_s=2.8,
    actor_deceleration_duration_s=4.8, final_actor_speed_mps=0.,
    target_lane_center_source_y_m=0., observation_end_s=18.8,
    direction='negative source-y, from ego left lane into ego lane',
    interpretation='scripted native-vehicle trajectory intervention, including gradual deceleration to stop; not reactive traffic',
    actor_motion_limits=dict(max_speed_mps=35., max_acceleration_mps2=8., max_yaw_rate_radps=.8),
    minimum_other_actor_gap_m=.2, no_new_actor=True, road_weather_others_unchanged=True)

def polygon(t,q,size):
    l,w,_=size
    corners=np.array([[-l/2,-w/2,0],[l/2,-w/2,0],[l/2,w/2,0],[-l/2,w/2,0]])
    return Polygon((Rotation.from_quat(q[[1,2,3,0]]).apply(corners)+t)[:,:2])

def generate(folder, baseline):
    folder=Path(folder);baseline=Path(baseline);folder.mkdir(parents=True,exist_ok=True)
    source=read(BUNDLE/'source_manifest.json');identity=read(source['instances_info.json']['path'])['5']
    if source['scene']!=551 or identity['id']!=CONFIG['annotation_id'] or identity['class_name']!='Vehicle':
        raise ValueError('Native scene/actor identity mismatch')
    with np.load(BUNDLE/'actors.npz') as z:raw={k:z[k].copy() for k in z.files}
    with np.load(baseline/'A/actor_tracks.npz') as z:effective={k:z[k].copy() for k in z.files}
    original_t=raw['RigidNodes__instances_trans'][:,3];original_q=raw['RigidNodes__instances_quats'][:,3]
    origin=np.loadtxt(Path(source['raw_data'])/'ego_pose/000.txt')
    annotation_pose=np.linalg.inv(origin)@np.asarray(identity['frame_annotations']['obj_to_world'][0])
    np.testing.assert_allclose(annotation_pose[:3,3],original_t[0],rtol=0,atol=1e-5)
    np.testing.assert_allclose(identity['frame_annotations']['box_size'][0],raw['RigidNodes__instances_size'][3],rtol=0,atol=1e-5)
    t=effective['RigidNodes__instances_trans'][:,3];q=effective['RigidNodes__instances_quats'][:,3]
    if len(t)!=199 or not effective['RigidNodes__instances_fv'][:,3].all():raise ValueError('Invalid native track support')
    times=np.arange(199)/10.;curve=CubicSpline(times,t,axis=0)
    rotation=Slerp(times,Rotation.from_quat(q[:,[1,2,3,0]]))
    start=CONFIG['absolute_source_activation_s']
    phases=validate_phases(CONFIG['phases']);timeline=Timeline(phases,start)
    duration=sum(p['duration_s'] for p in phases)
    def evaluate(clock):
        source_time,weight=timeline.evaluate(clock)
        # Out-of-support proposals are rejected below; clipping only allows an auditable rejection artifact.
        base=curve(np.clip(source_time,times[0],times[-1]));v=base.copy();v[:,1]*=(1-weight)
        return v,np.clip(source_time,times[0],times[-1])
    positions,retime=evaluate(times)
    dense=np.arange(CONFIG['initial_source_frame']/10.,19.80001,.01)
    dense_pos,_=evaluate(dense);velocity=np.gradient(dense_pos,.01,axis=0)
    acceleration=np.gradient(velocity,.01,axis=0)
    # Analytic positions sampled on either side give stable heading up to the stop.
    d=.0001;vel=(evaluate(np.minimum(times+d,19.8))[0]-evaluate(np.maximum(times-d,0))[0])/(2*d)
    tangent=curve(retime,1);heading=np.arctan2(vel[:,1],vel[:,0]);speed=np.linalg.norm(vel[:,:2],axis=1)
    for i in range(1,len(times)):
        if speed[i]<1e-5:heading[i]=heading[i-1]
    delta=heading-np.arctan2(tangent[:,1],tangent[:,0])
    rotations=Rotation.from_rotvec(np.c_[np.zeros(199),np.zeros(199),delta])*rotation(retime)
    quats=rotations.as_quat()[:,[3,0,1,2]]
    # Preserve checkpoint values exactly; renderer retains its heldout interpolation before activation.
    prefix=times<=start+1e-9;positions[prefix]=original_t[prefix];quats[prefix]=original_q[prefix]
    rendered_t=positions.copy();rendered_q=quats.copy();rendered_t[prefix]=t[prefix];rendered_q[prefix]=q[prefix]
    yaw=np.unwrap(Rotation.from_quat(rendered_q[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    metrics=dict(max_speed_mps=float(np.linalg.norm(velocity[:,:2],axis=1).max()),
        max_acceleration_mps2=float(np.linalg.norm(acceleration[:,:2],axis=1).max()),
        max_yaw_rate_radps=float(np.max(np.abs(np.diff(yaw[10:]))/.1)))
    phase_motion=[];offset=start
    for index,phase in enumerate(phases,1):
        mask=(dense>=offset)&(dense<=offset+phase['duration_s'])
        if mask.any():phase_motion.append(dict(phase_index=index,action=phase['action'],max_speed_mps=float(np.linalg.norm(velocity[mask,:2],axis=1).max()),max_acceleration_mps2=float(np.linalg.norm(acceleration[mask,:2],axis=1).max())))
        offset+=phase['duration_s']
    # Audit the actual renderer/collision interpolation, at 10 substeps per source frame,
    # against ALL other reconstructed participant groups, not an assumed future ego.
    conflicts=[];minimum=float('inf');closest=None
    group_data=[]
    for group in ['RigidNodes','DeformableNodes','SMPLNodes']:
        trans=effective[group+'__instances_trans'];quat=effective[group+'__instances_quats']
        if quat.ndim==4:quat=quat[:,:,0,:]
        group_data.append((group,trans,quat,effective[group+'__instances_size'],effective[group+'__instances_fv']))
    size=effective['RigidNodes__instances_size'][3]
    for frame in np.arange(10.,198.0001,.1):
        lo=int(np.floor(frame+1e-8));hi=min(lo+1,198);u=frame-lo
        aq=Slerp([0.,1.],Rotation.from_quat(rendered_q[[lo,hi]][:,[1,2,3,0]]))([max(0.,u)])[0].as_quat()[[3,0,1,2]]
        at=rendered_t[lo]*(1-u)+rendered_t[hi]*u;a=polygon(at,aq,size)
        for group,trans,quat,sizes,valid in group_data:
            for other in np.flatnonzero(valid[lo]|valid[hi]):
                if group=='RigidNodes' and other==3:continue
                if valid[lo,other] and valid[hi,other]:
                    bt=trans[lo,other]*(1-u)+trans[hi,other]*u
                    bq=Slerp([0.,1.],Rotation.from_quat(quat[[lo,hi],other][:,[1,2,3,0]]))([max(0.,u)])[0].as_quat()[[3,0,1,2]]
                else:
                    j=lo if valid[lo,other] else hi;bt=trans[j,other];bq=quat[j,other]
                if abs(bt[2]-at[2])>(size[2]+sizes[other,2])/2:continue
                gap=float(a.distance(polygon(bt,bq,sizes[other])))
                if gap<minimum:minimum=gap;closest=dict(source_frame=float(frame),group=group,index=int(other))
                if gap<CONFIG['minimum_other_actor_gap_m']:conflicts.append(dict(source_frame=float(frame),group=group,index=int(other),gap_m=gap))
    limits=CONFIG['actor_motion_limits']
    checks={k:metrics[k]<=v for k,v in limits.items()}
    checks.update(prefix_bitwise_unchanged=np.array_equal(positions[prefix],original_t[prefix]) and np.array_equal(quats[prefix],original_q[prefix]),
        annotation_to_model_identity_verified=True,
        no_other_actor_overlap_or_close_pass=not conflicts,rotation_finite=bool(np.isfinite(quats).all()),
        action_sequence_valid=True,source_clock_within_support=bool(timeline.evaluate(dense)[0].max()<=times[-1]+1e-9),positions_finite=bool(np.isfinite(positions).all()))
    a=read(baseline/'A/run_report.json');initial=np.asarray(a['observations'][0]['pose'])
    relative=(np.linalg.inv(initial)@np.r_[t[10],1])[:3]
    np.savez_compressed(folder/'target_track.npz',original_trans=original_t,original_quats=original_q,
        edited_trans=positions,edited_quats=quats,rendered_trans=rendered_t,rendered_quats=rendered_q,times=times)
    report=dict(config=CONFIG,passed=all(checks.values()),checks=checks,metrics=metrics,
        source_scene=asset(BUNDLE/'source_manifest.json'),source_actors=asset(BUNDLE/'actors.npz'),source_identity=source['instances_info.json'],
        annotation_to_model_initial_center_error_m=float(np.linalg.norm(annotation_pose[:3,3]-original_t[0])),
        baseline_effective_tracks=asset(baseline/'A/actor_tracks.npz'),initial_target_relative_forward_left_up_m=relative.tolist(),
        onset_target_source_xyz_m=evaluate([start])[0][0].tolist(),
        final_target_source_xyz_m=positions[-1].tolist(),actor_speed_at_onset_mps=float(np.linalg.norm(curve(start,1)[:2])),
        action_sequence_complete_local_s=CONFIG['intervention_start_s']+duration,
        action_program='native551_car3_action_sequence',action_phases=phases,phase_motion=phase_motion,
        minimum_other_actor_gap_m=minimum if np.isfinite(minimum) else None,closest_other_actor=closest,conflicts=conflicts,
        audit_substep_s=.01,audit_source_frames=[10,198],audit_against_all_native_participant_groups=True,
        actor_height_source='retimed source road profile; target lane contact not independently certified',
        ego_future_used_online=False,ego_gap_and_response_require_actual_closed_loop=True,gpu_render_sync_verified=False)
    write(folder/'scenario.json',report)
    if not report['passed']:raise ValueError('Cut-in motion/other-actor audit failed: '+str(folder/'scenario.json'))
    # Offline design picture. Only A has a measured ego path until B is actually run.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    a_pose=np.asarray([o['pose'] for o in a['observations']])
    axes[0].plot(a_pose[:,0,3],a_pose[:,1,3],label='A actual ego')
    axes[0].plot(t[10:,0],t[10:,1],'--',label='A original native3')
    axes[0].plot(rendered_t[10:,0],rendered_t[10:,1],label='B scripted native3 (offline design)')
    axes[0].set(xlim=(10,125),ylim=(-3,9),xlabel='Source forward x (m)',ylabel='Source left y (m)');axes[0].legend(fontsize=8)
    axes[1].plot(dense-1,np.linalg.norm(velocity[:,:2],axis=1),label='B target speed (m/s)')
    axes[1].plot(dense-1,np.linalg.norm(acceleration[:,:2],axis=1),label='B target acceleration (m/s2)')
    axes[1].set(xlabel='Local simulation time (s)',ylabel='Offline scripted motion');axes[1].legend(fontsize=8)
    fig.suptitle('OFFLINE DESIGN ONLY: B ego response and camera effect require actual GPU closed loop')
    fig.tight_layout();fig.savefig(folder/'trajectory_design.png',dpi=150);plt.close(fig)
    return report
