"""Edit design only: one original car overtakes, merges, and brakes."""
import hashlib,json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from paths import HERE,SCENE_BUNDLE,RAW

def smooth(u):return 10*u**3-15*u**4+6*u**5
def integral(u):return 2.5*u**4-3*u**5+u**6

def compile_vehicle_path(path,original_trans,original_quats,fps=10.):
 from actor_trajectories import validate_path
 value=validate_path(path)
 if value['coordinate_frame']!='source_world' or value.get('motion_profile')!='smooth_v1' or value.get('heading_mode')!='path_tangent':raise ValueError('Rigid vehicles require world-frame C2 smooth path tangent')
 knots=np.array([r['timestamp_s'] for r in value['waypoints']]);points=np.array([r['position_world_m'] for r in value['waypoints']]);times=np.arange(len(original_trans))/fps
 if knots[0]!=0. or not np.isclose(knots[-1],times[-1]) or np.max(np.abs(knots*fps-np.rint(knots*fps)))>1e-6:raise ValueError('Vehicle path must cover whole source window on source clock')
 if not np.allclose(points[0],original_trans[0],rtol=0,atol=1e-6):raise ValueError('Initial native pose must remain unchanged')
 curve=CubicSpline(knots,points,axis=0);dense=np.linspace(knots[0],knots[-1],max(2,int(knots[-1]*100)+1));v=curve(dense,1);a=curve(dense,2)
 speed=np.linalg.norm(v,axis=1);yaw=np.unwrap(np.arctan2(v[:,1],v[:,0]));rate=np.abs(np.diff(yaw)/np.diff(dense))
 if not np.isfinite(points).all() or speed.min()<.05 or speed.max()>35 or np.linalg.norm(a,axis=1).max()>12 or rate.max()>.8:raise ValueError('Vehicle trajectory exceeds declared diagnostic motion budget')
 trans=curve(times);trans[0]=original_trans[0];velocity=curve(times,1);head=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]));old=Rotation.from_quat(np.asarray(original_quats)[:,[1,2,3,0]]);oldyaw=old.as_euler('xyz')[:,2];delta=head-oldyaw;delta[0]=0.;q=(Rotation.from_rotvec(np.c_[np.zeros(len(delta)),np.zeros(len(delta)),delta])*old).as_quat()[:,[3,0,1,2]];q[0]=original_quats[0]
 return trans,q,dict(max_speed_mps=float(speed.max()),max_acceleration_mps2=float(np.linalg.norm(a,axis=1).max()),max_yaw_rate_radps=float(rate.max()),source_time_only=True)

def generate():
 with np.load(SCENE_BUNDLE/'actors.npz') as z:t=z['RigidNodes__instances_trans'][:,3].copy();q=z['RigidNodes__instances_quats'][:,3].copy();visible=z['RigidNodes__instances_fv'][:,3].copy()
 if not visible.all():raise ValueError('Chosen original vehicle lacks complete source visibility')
 times=np.arange(199)/10.;anchor=t[0];v0=float((t[5,0]-t[0,0])/.5);v1=4.;start=.2;duration=4.2;u=np.clip((times-start)/duration,0,1)
 distance=v0*times+(v1-v0)*duration*integral(u)+(v1-v0)*np.maximum(times-start-duration,0)
 lateral=anchor[1]+(.3-anchor[1])*smooth(np.clip((times-.6)/2.8,0,1));x=anchor[0]+distance
 # Height follows the original car's reconstructed road profile at the NEW x.
 order=np.argsort(t[:,0]);height=np.interp(x,t[order,0],t[order,2]);height[0]=anchor[2];position=np.c_[x,lateral,height]
 ids=np.arange(0,199,2);path=dict(schema='actor_trajectory_v1',coordinate_frame='source_world',outside_interval='hold',motion_profile='smooth_v1',heading_mode='path_tangent',waypoints=[dict(timestamp_s=float(times[i]),position_world_m=position[i].tolist(),yaw_delta_degrees=0.) for i in ids])
 trans,quats,audit=compile_vehicle_path(path,t,q)
 config=dict(schema='native_vehicle_cutin551_v1',scene=551,start_frame=0,controls=100,initial_speed_mps=8.,fps=10.,native_actor='native:RigidNodes:3',source_instance_index=5,source_annotation_id='o24FGKGBO7aPlRVYkDyU7A',no_managed_vehicle_inserted=True,trajectory=path,nominal_merge_interval_s=[.6,3.4],braking_interval_s=[.2,4.4],final_actor_speed_mps=v1,evaluation_plan_horizon_s=2.5,source_nn_horizon_s=3.,evaluation_plan_dt_s=.5,wall_cap_seconds=14400,kinematic_audit=audit)
 (HERE/'scenario.json').write_text(json.dumps(config,indent=2)+'\n');np.savez_compressed(HERE/'designed_actor_track.npz',positions=trans,quaternions_wxyz=quats,original_positions=t,original_quaternions_wxyz=q,visible=visible)
 print(json.dumps({k:v for k,v in config.items() if k!='trajectory'},indent=2));return config
if __name__=='__main__':
 import sys
 from paths import PHASE26
 sys.path.insert(0,str(PHASE26/'bridge/framework_backend'));generate()
