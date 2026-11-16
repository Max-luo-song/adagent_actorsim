"""Time-local forward-motion tube, qualified by current multi-frame evidence.

This declares a forward/non-reversing acceleration model, just as the prior
stationary-to-reported-velocity capsule declared a non-reversing straight model.
It is not a physical safety certificate. Missing or conflicting evidence keeps
that prior conservative model. Detector score alone never qualifies a tube.
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

import copy,json
pass  # Path is provided by the portable boundary above
import numpy as np
from planning_guard import sample_half_intervals,rectangle_axes
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name('motion_tube_policy.json').read_text()))

def qualify(history,velocities,protected,now,cfg,rotation):
 if len(history)<POLICY['minimum_observations'] or len(protected)<2 or not all(protected[-2:]):return None
 recent=history[-max(3,int(cfg.get('confirmation_observations',3))):];times=np.array([x[0] for x in recent]);points=np.asarray([x[1] for x in recent]);reported=np.asarray(velocities)[-len(recent):,:2]
 if len(reported)!=len(recent) or np.ptp(times)<=0 or np.any(np.diff(times)<=0):return None
 velocity=np.median(reported,axis=0);speed=float(np.linalg.norm(velocity));margin=float(cfg.get('obstacle_margin_m',.5));lease=float(cfg.get('max_gap_seconds',.75))
 if not np.isfinite(points).all() or not np.isfinite(reported).all() or speed<=margin/lease:return None
 direction=velocity/speed;normal=np.array([direction[1],-direction[0]]);design=np.c_[times-times[-1],np.ones(len(times))];fit=np.linalg.lstsq(design,points[:,:2],rcond=None)[0];residual=float(np.linalg.norm(points[:,:2]-design@fit,axis=1).max());forward=float(fit[0]@direction)
 if residual>margin or forward<=0:return None
 disagreement=fit[0]-velocity;floor=margin/lease;long_error=max(abs(float(disagreement@direction)),float(np.max(np.abs((reported-velocity)@direction))),floor);side_error=max(abs(float(disagreement@normal)),float(np.max(np.abs((reported-velocity)@normal))),floor)
 # The spread between independent estimates is already represented by
 # their hull. Expanding that hull by the same spread a second time invents
 # an extra slow/fast branch. Keep EVERY reported sample (not only the median)
 # and the position-fit estimate, then apply the existing metric floor once.
 # This remains an empirical prediction interval, not a certified speed bound.
 sources=np.r_[reported@direction,forward]
 low=float(np.min(sources)-floor);high=float(np.max(sources)+floor)
 if low<=floor or high<=low or not np.isfinite([low,high,side_error]).all():return None
 local=(np.asarray(rotation).T@np.r_[direction,0.])[:2];local/=np.linalg.norm(local)
 return dict(schema='qualified_time_local_motion_v1',timestamp_s=float(now),direction_xy=local.tolist(),speed_lower_mps=low,speed_upper_mps=high,transverse_velocity_error_mps=side_error,observations=len(recent),window_s=float(np.ptp(times)),position_fit_residual_m=residual,depth_supported_observations=2,uncertainty_is_certified=False,actor_acceleration_budget_mps2=POLICY['actor_acceleration_budget_mps2'],integration_dt_s=float(np.min(np.diff(times))),velocity_interval_policy='source_hull_plus_metric_floor_v2',longitudinal_velocity_sources_mps=sources.tolist(),velocity_interval_padding_mps=float(floor))

def travel_lower(speed,time,acceleration,dt=.1):
 # Semi-implicit Euler is below the continuous braking distance. Enclose both
 # physical continuous motion and the simulator's velocity-first integration.
 time=np.asarray(time,dtype=float);n=np.floor(time/dt+1e-10);m=np.minimum(n,np.maximum(0.,np.ceil(speed/(acceleration*dt)-1e-12)-1.));distance=dt*(m*speed-acceleration*dt*m*(m+1.)/2.)
 fraction=np.maximum(0.,time-n*dt);return np.maximum(0.,distance+fraction*np.maximum(speed-acceleration*dt*(n+1.),0.))

def tube_boxes(box,certificate,times,radius=None):
 times=np.asarray(times,dtype=float);half=sample_half_intervals(times);lo=np.maximum(0.,times-half);hi=times+half;a=float(certificate['actor_acceleration_budget_mps2']);direction=np.asarray(certificate['direction_xy']);radius=max(float(radius or 0.),float(np.hypot(*np.asarray(box)[3:5])/2));dt=float(certificate['integration_dt_s'])
 lower=travel_lower(certificate['speed_lower_mps'],lo,a,dt);upper=certificate['speed_upper_mps']*hi+.5*a*hi**2+.5*a*hi*dt;side=certificate['transverse_velocity_error_mps']*hi+.5*a*hi**2+.5*a*hi*dt
 centers=np.asarray(box)[:2]+((lower+upper)/2)[:,None]*direction;boxes=np.tile(np.asarray(box,dtype=float),(len(times),1));boxes[:,:2]=centers;boxes[:,3]=2*(radius+side);boxes[:,4]=2*radius+upper-lower;boxes[:,6]=np.arctan2(direction[0],direction[1]);boxes[:,7:9]=0.
 return boxes

def paired_rectangle_gap(states,boxes,half_width,half_length):
 er,ef=rectangle_axes(states[:,2]);br,bf=rectangle_axes(boxes[:,6]);axes=np.stack([er,ef,br,bf],axis=1);delta=boxes[:,:2]-states[:,:2];projection=np.abs(np.sum(delta[:,None]*axes,axis=-1));ego=half_width*np.abs(np.sum(er[:,None]*axes,axis=-1))+half_length*np.abs(np.sum(ef[:,None]*axes,axis=-1));object_radius=boxes[:,3,None]/2*np.abs(np.sum(br[:,None]*axes,axis=-1))+boxes[:,4,None]/2*np.abs(np.sum(bf[:,None]*axes,axis=-1));return np.max(projection-ego-object_radius,axis=1)

from expired_memory_guard import PlanningGuard as PreviousGuard
class PlanningGuard(PreviousGuard):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.local_motion={};report=args[0] if args else kwargs.get('report',{});now=report.get('global_evidence_receipt',{}).get('timestamp');obstacles={o['candidate_index']:o for o in report.get('obstacles',[])}
  self.motion_lease=float(report.get('config',{}).get('max_gap_seconds',.75))
  if not np.isfinite(self.motion_lease) or self.motion_lease<=0:raise ValueError('Invalid motion observation lease')
  for index,row in enumerate(r for r in self.rows if r['hard_constraint']):
   obstacle=obstacles[row['candidate_index']];cert=obstacle.get('geometry_time_local_motion')
   if not cert or cert.get('schema')!='qualified_time_local_motion_v1' or obstacle['state']=='last_known_confirmed' or cert.get('timestamp_s')!=now:continue
   d=np.asarray(cert.get('direction_xy'),dtype=float);values=np.asarray([cert.get('speed_lower_mps'),cert.get('speed_upper_mps'),cert.get('transverse_velocity_error_mps'),cert.get('actor_acceleration_budget_mps2'),cert.get('integration_dt_s')],dtype=float)
   if d.shape!=(2,) or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1.)>1e-6 or not np.isfinite(values).all() or values[0]<=0 or values[1]<values[0] or values[2]<0 or values[3]<=0 or values[4]<=0:raise ValueError('Invalid qualified motion tube')
   # Conflicting current motion must retain the inherited conservative shape.
   velocity=np.asarray(obstacle['box'],dtype=float)[7:9]
   if obstacle.get('declared_static_epoch') is not None or float(velocity@d)<=0:continue
   self.local_motion[index]=cert;row.update(geometry='time_local_forward_motion_tube',motion_tube=cert,previous_geometry=row['geometry'])
 def clearance(self,states,times,_full_motion=False):
  base_clearance=PreviousGuard.full_clearance if _full_motion else PreviousGuard.clearance
  if not self.local_motion:return base_clearance(self,states,times)
  states=np.asarray(states);times=np.asarray(times)
  if len(states)!=len(times) or not np.isfinite(times).all() or np.any(times<0):raise ValueError('Invalid motion query times')
  keep=np.array([i for i in range(len(self.hard_boxes)) if i not in self.local_motion],dtype=int);other=copy.copy(self)
  for name in ('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges'):setattr(other,name,getattr(self,name)[keep])
  result=base_clearance(other,states,times)
  for index,cert in self.local_motion.items():
   boxes=tube_boxes(self.hard_boxes[index],cert,times,self.hard_radii[index]);gap=paired_rectangle_gap(states,boxes,self.vehicle.half_width_m,self.vehicle.half_length_m)-self.hard_margins[index]
   # Current observation uses the exact inherited shape; only future motion changes.
   initial=np.flatnonzero(times==0.)
   if len(initial):
    single=copy.copy(self)
    for name in ('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges'):setattr(single,name,getattr(self,name)[index:index+1])
    gap[initial]=PreviousGuard.clearance(single,states,times)[initial]
   if not _full_motion:gap[times>self.motion_lease+1e-9]=np.inf
   result=np.minimum(result,gap)
  return result
 def full_clearance(self,states,times):
  return self.clearance(states,times,_full_motion=True)
 def soft_cost(self,states,times):
  cost=super().soft_cost(states,times)
  if not self.local_motion:return cost
  times=np.asarray(times);future=times>self.motion_lease+1e-9
  if not np.any(future):return cost
  gap=self.full_clearance(np.asarray(states),times)[future];penetration=np.maximum(-gap,0.);return float(cost+np.mean((penetration/(1.+penetration))**2))
 def audit(self):
  result=super().audit();result['time_local_motion']=dict(qualified_targets=len(self.local_motion),policy=POLICY,current_hard_motion_lease_s=self.motion_lease,far_uncertainty='bounded risk with full-envelope current-action braking viability; not a confirmed stationary wall',complete_envelope_retained_for_braking=True,stationary_and_unqualified_targets_use_original_geometry=True,raw_candidates_removed=False,safety_margins_reduced=False);return result
