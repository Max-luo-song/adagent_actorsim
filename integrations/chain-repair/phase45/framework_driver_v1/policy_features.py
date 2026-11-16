"""Local planning descriptors; no scene/frame IDs, labels or future truth."""
import hashlib,json
import numpy as np
LONGITUDINAL=np.array([0.,2.,4.,6.,8.,10.,12.,16.,20.,24.,28.,32.,40.,48.])
LATERAL=np.array([-8.,-6.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,6.,8.])
FEATURE_SCHEMA='corridor_clearance_road_reference_state_v2'
def encode(problem):
 p=problem;v=p.vehicle;xx,yy=np.meshgrid(LATERAL,LONGITUDINAL,indexing='xy');q=np.tile(p.initial,(xx.size,1));q[:,:2]+=np.c_[xx.ravel(),yy.ravel()];maps=[]
 from empty_controller import sweep_allowance
 for tau in (0.,.5,1.,2.):
  gap=np.empty(len(q));clock=tau+LONGITUDINAL/max(float(p.initial[3]),.5)
  for column in range(len(LATERAL)):
   ids=np.arange(column,len(q),len(LATERAL));query=q[ids];gap[ids]=np.minimum(p.temporal.clearance(query,clock),p.guard.clearance(query,clock))-sweep_allowance(query,v)
  maps.append(np.clip(gap,-4.,12.)/12.)
 road=np.full(len(q),-4.) if p.road is None else p.road.clearance(q);maps.append(np.clip(road,-4.,12.)/12.)
 from scipy.spatial import cKDTree
 ambiguous=np.asarray(p.reliability.get('observed_depth_geometry',{}).get('height_ambiguity_braking_points_xy',[]),dtype=float).reshape(-1,2);epistemic=np.asarray(p.aligned.epistemic.points,dtype=float).reshape(-1,2)
 for points in (ambiguous,epistemic):
  distance=cKDTree(points).query(q[:,:2])[0] if len(points) else np.full(len(q),12.);maps.append(np.clip(distance,0.,12.)/12.)
 cap=p.guard.speed_limit
 scalar=np.r_[p.initial[3]/v.max_speed_mps,p.initial[4]/np.deg2rad(v.max_steer_degrees),p.aligned.cruise_preference/v.max_speed_mps,(cap or 0.)/v.max_speed_mps,float(cap is not None),float(p.aligned.stop),p.offset/3.,p.reference[:,0]/25.,p.reference[:,1]/50.,float(p.reliability.get('retained_forecast') is not None),float(bool(p.reliability.get('perception_degraded'))),float(bool(p.reliability.get('need_braking_fallback'))),p.interval/2.,np.log1p(len(ambiguous))/10.,np.log1p(len(epistemic))/10.,np.log1p(getattr(p.guard,'expired_memory_count',0))/10.]
 result=np.r_[scalar,*maps].astype(np.float32)
 if not np.isfinite(result).all():raise ValueError('Nonfinite student features')
 return result
CONTROL_FEATURES=27
def identity(features):return hashlib.sha256(np.asarray(features,dtype=np.float32).tobytes()).hexdigest()
def vehicle_identity(vehicle):
 return hashlib.sha256(json.dumps(vars(vehicle),sort_keys=True).encode()).hexdigest()
def mode_of(row,reference,vehicle):
 lidar=np.asarray(row['lidar']);stride=int(round(.5/vehicle.dt));residual=float(np.mean(lidar[stride:6*stride+1:stride,0]-np.asarray(reference)[:,0]));threshold=vehicle.half_width_m*.5
 return 1 if residual < -threshold else 2 if residual > threshold else 0
