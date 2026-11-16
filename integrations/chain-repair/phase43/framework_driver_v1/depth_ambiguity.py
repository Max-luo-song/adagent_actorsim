"""One-step braking viability for height-ambiguous observed surfaces.

Uncertain points remain observed risk; they are not erased or called free road.
Before executing each fresh 0.1s action, preserve a bounded stop with the same
body/voxel margins. Unexecuted six-second plans need not pretend these samples
are confirmed stationary objects. Loss of observations uses immediate fallback.
"""
import numpy as np
from temporal_context import FreshGeometry
from mpc_perf_v1.exact_memo import Memo
class ObservationViability:
 def __init__(self,report,info,vehicle,motion_guard=None):
  self.known_clear=None
  self.vehicle=vehicle;meta=report.get('observed_depth_geometry',{});self.points=np.asarray(meta.get('height_ambiguity_braking_points_xy',[]),dtype=float).reshape(-1,2)
  if not np.isfinite(self.points).all():raise ValueError('Invalid ambiguous depth geometry')
  self.motion_active=bool(motion_guard is not None and getattr(motion_guard,'local_motion',{}))
  self.memory_active=bool(motion_guard is not None and getattr(motion_guard,'expired_memory_count',0))
  self.active=bool(len(self.points)) or self.motion_active or self.memory_active;self.reaction=float(report.get('config',{}).get('confirmation_seconds',.2))
  if not np.isfinite(self.reaction) or self.reaction<=0:raise ValueError('Invalid ambiguity reaction interval')
  self.geometry=FreshGeometry(vehicle,observed_points=self.points,observed_point_radius=float(meta.get('voxel_half_diagonal_m',np.sqrt(3)*.25)),observed_cell_rotation=info['explicit_l2g_r_mat'] if meta.get('policy')=='observed_depth_geometry_with_free_space_clearing_v1' else None)
  self.memo=Memo(capacity=128);self.gap=self.memo.wrap(self._gap)
 def clearance(self,states):
  from empty_controller import sweep_allowance
  return self.geometry.clearance(states)-sweep_allowance(states,self.vehicle)
 def stop_states(self,initial,action,mode='hold'):
  from empty_controller import integrate_state
  v=self.vehicle;initial=np.asarray(initial,dtype=float);action=np.asarray(action,dtype=float)
  rows=[initial.copy(),integrate_state(initial,action,v.dt,v.wheelbase_m)]
  if rows[-1][3]<-1e-5:return np.asarray(rows)
  count=int(np.ceil(max(0.,rows[-1][3])/v.emergency_braking_mps2/v.dt))+int(np.ceil(self.reaction/v.dt))
  for k in range(count):
   previous=rows[-1];acc=0. if k*v.dt<self.reaction-1e-9 else -min(v.emergency_braking_mps2,max(0.,previous[3])/v.dt)
   rate=0.
   if k*v.dt>=self.reaction-1e-9 and mode!='hold' and previous[3]>1e-8:
    from native_contract import CONTRACT
    bounds=CONTRACT['comfort_boundaries'];angle=np.deg2rad(v.max_steer_degrees);coupled=min(angle,np.arctan(bounds['max_abs_lat_accel']*v.wheelbase_m/max(previous[3]**2,1e-9)),np.arctan(bounds['max_abs_yaw_rate']*v.wheelbase_m/max(previous[3],1e-9)))
    target=0. if mode=='unwind' else (-coupled if mode=='left' else coupled);rate=float(np.clip((target-previous[4])/v.dt,-np.deg2rad(v.max_steer_rate_degrees),np.deg2rad(v.max_steer_rate_degrees)))
   rows.append(integrate_state(previous,[acc,rate],v.dt,v.wheelbase_m))
  return np.asarray(rows)
 def _gap(self,initial,action):
  if not self.active:return float('inf')
  gaps=[]
  for mode in ['hold','unwind','left','right']:
   states=self.stop_states(initial,action,mode);clearance=self.clearance(states)
   if self.known_clear is not None:clearance=np.minimum(clearance,self.known_clear(states))
   gaps.append(float(clearance.min()))
  return max(gaps)
 def audit(self):return dict(policy='current_action_preserves_bounded_stop_v2',enabled=self.active,points=len(self.points),motion_uncertainty_active=self.motion_active,expired_memory_backup_active=self.memory_active,reaction_seconds=self.reaction,braking_witness_modes=['hold','unwind','left','right'],execution_seconds=self.vehicle.dt,observation_rechecked_every_step=True,points_erased=False,free_space_asserted=False,whole_future_prediction_certified=False)

# Existing depth-only callers keep the same API.
DepthAmbiguity=ObservationViability
