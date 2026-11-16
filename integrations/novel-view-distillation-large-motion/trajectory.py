"""One visible lane displacement with heading derived from actual path tangent."""
import numpy as np

POLICY=dict(left_displacement_m=4.,onset_s=.5,duration_s=3.,
    heading='path tangent projected onto the verified gravity plane',
    recorded_longitudinal_progress_preserved=True,recorded_actor_clock_preserved=True,
    controlled_driving=False,road_compliance_certified=False)

def build(compiled):
    samples=compiled['samples'];times=np.array([s['time_s'] for s in samples]);recorded=np.array([s['recorded_ego_to_world'] for s in samples],float)
    up=np.asarray(compiled['gravity_up_source'],float);up/=np.linalg.norm(up)
    forward=recorded[:,:3,0];forward=forward-np.einsum('ij,j->i',forward,up)[:,None]*up;forward/=np.linalg.norm(forward,axis=1)[:,None]
    left=np.cross(up,forward);u=np.clip((times-POLICY['onset_s'])/POLICY['duration_s'],0,1)
    offsets=POLICY['left_displacement_m']*(10*u**3-15*u**4+6*u**5)
    xyz=recorded[:,:3,3]+left*offsets[:,None]
    velocity=np.gradient(xyz,times,axis=0,edge_order=2)
    tangent=velocity-np.einsum('ij,j->i',velocity,up)[:,None]*up
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    heading_left=np.cross(up,tangent)
    poses=np.repeat(np.eye(4)[None],len(times),axis=0)
    poses[:,:3,:3]=np.stack([tangent,heading_left,np.repeat(up[None],len(times),axis=0)],axis=-1);poses[:,:3,3]=xyz
    delta=np.linalg.inv(recorded)@poses
    yaw=np.degrees(np.arctan2(delta[:,1,0],delta[:,0,0]))
    acceleration=np.gradient(velocity,times,axis=0,edge_order=2)
    checks=dict(initial_position_unchanged=np.linalg.norm(xyz[0]-recorded[0,:3,3])<1e-6,
        final_left_at_least_39m=delta[-1,1,3]>=3.9,
        maximum_relative_heading_at_least_5deg=np.max(abs(yaw))>=5,
        all_poses_rigid=np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-6),
        all_camera_horizons_use_same_gravity=np.allclose(poses[:,:3,2],up,atol=1e-6),
        heading_matches_path_tangent=np.allclose(poses[:,:3,0],tangent,atol=1e-6),
        no_position_jump=np.max(np.linalg.norm(np.diff(xyz,axis=0),axis=1))<3.,
        maximum_speed_below_25mps=np.max(np.linalg.norm(velocity,axis=1))<25.,
        maximum_acceleration_below_5mps2=np.max(np.linalg.norm(acceleration,axis=1))<5.)
    checks={k:bool(v) for k,v in checks.items()}
    if not all(checks.values()):raise ValueError('Large-motion trajectory checks failed: '+str(checks))
    task=dict(schema='ego_trajectory_v1',coordinate_frame='source_world',start_frame=compiled['start_frame'],attitude_policy='gravity_level',
        waypoints=[dict(time_s=float(t),ego_to_world=p.tolist()) for t,p in zip(times,poses)])
    audit=dict(policy=POLICY,checks=checks,duration_s=float(times[-1]),samples=len(times),
        maximum_relative_heading_deg=float(np.max(abs(yaw))),final_left_m=float(delta[-1,1,3]),
        maximum_speed_mps=float(np.max(np.linalg.norm(velocity,axis=1))),
        maximum_acceleration_mps2=float(np.max(np.linalg.norm(acceleration,axis=1))),
        rows=[dict(source_frame=s['source_frame'],time_s=float(t),left_m=float(d[1,3]),forward_delta_m=float(d[0,3]),relative_yaw_deg=float(y)) for s,t,d,y in zip(samples,times,delta,yaw)])
    return task,audit
