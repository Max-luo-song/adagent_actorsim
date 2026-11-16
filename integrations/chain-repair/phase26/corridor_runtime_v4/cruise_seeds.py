"""Smooth symmetric forward corridors, fitted to the actual bounded bicycle model."""
import numpy as np
from scipy.optimize import least_squares
from corridor_geometry import POLICY

def propose(initial,vehicle,guard,temporal,horizon,interval,interpolation='constant'):
 from empty_controller import rollout
 speed=float(initial[3]);margin=vehicle.clearance_margin_m;body=vehicle.half_width_m;offsets={-1:-(2*body+2*margin),1:2*body+2*margin};nearest=np.inf;release_y=float(initial[1]+max(speed,1.)*horizon+vehicle.half_length_m)
 for box,radius,extra in zip(guard.hard_boxes,guard.hard_radii,guard.hard_margins):
  if box[1]<=initial[1] or box[1]-initial[1]>max(speed*horizon,vehicle.half_length_m):continue
  width=float(radius)+float(extra)+body+margin
  if abs(box[0]-initial[0])>width:continue
  offsets[-1]=min(offsets[-1],float(box[0]-initial[0]-width));offsets[1]=max(offsets[1],float(box[0]-initial[0]+width));nearest=min(nearest,float(box[1]-initial[1]-radius-vehicle.half_length_m));release_y=max(release_y,float(box[1]+radius+vehicle.half_length_m+margin))
 anchor=min(horizon,POLICY['minimum_shift_time_s'],nearest/max(speed,1.))
 anchor=max(vehicle.dt,float(anchor));clock=np.arange(int(round(horizon/vehicle.dt))+1)*vehicle.dt;u=np.clip(clock/anchor,0.,1.);shape=10*u**3-15*u**4+6*u**5;derivative=30*u**2*(1-u)**2/anchor
 rows=[];specs={};rate=np.deg2rad(vehicle.max_steer_rate_degrees);offset=-np.asarray(initial[:2]);anchor_index=int(round(anchor/vehicle.dt))
 for braking in [False,True]:
  for sign in [-1,1]:
   name=('lane_change_' if braking else 'escape_')+('left' if sign<0 else 'right');displacement=offsets[sign];acc=np.zeros(6)
   if braking:acc[:2]=-min(1.,speed/max(2*interval,vehicle.dt))
   controls=np.c_[acc,np.zeros(6)];target_x=initial[0]+displacement*shape
   base_speed=np.maximum(.1,speed+np.minimum(clock,2*interval)*acc[0]);target_yaw=np.arctan2(displacement*derivative,base_speed)
   def residual(rates):
    controls[:,1]=rates;states,_=rollout(controls,speed,float(initial[4]),offset,vehicle.dt,vehicle.wheelbase_m,interval,interpolation)
    return np.r_[states[:,0]-target_x,2.*(states[:,2]-target_yaw),3.*states[-1,2],states[-1,4]]
   fit=least_squares(residual,np.zeros(6),bounds=(-rate,rate),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9);controls[:,1]=fit.x
   rows.append((name,controls.copy()));specs[name]=dict(sign=sign,displacement_m=displacement,anchor_index=anchor_index,anchor_time_s=anchor_index*vehicle.dt,release_forward_m=release_y,seed_fit_residual=float(np.linalg.norm(fit.fun)),seed_fit_success=bool(fit.success))
 return rows,dict(policy='smooth_forward_side_corridors_v1',lateral_offsets_m=[offsets[-1],offsets[1]],duration_s=anchor,final_trajectory_not_prescribed=False,side_preservation_required=True,corridors=specs,ground_truth_used=False)
