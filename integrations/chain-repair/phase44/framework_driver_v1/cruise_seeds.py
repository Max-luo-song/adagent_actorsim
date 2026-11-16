"""Symmetric cruising detours from current geometry, with no scene/ID rule."""
import numpy as np

def propose(initial,vehicle,guard,temporal,horizon,interval):
 speed=float(initial[3]);margin=vehicle.clearance_margin_m;body=vehicle.half_width_m;required=2*body+2*margin;offsets={-1:-required,1:required};nearest=np.inf
 # Only hypotheses intersecting the present forward corridor set the lateral
 # starts. They remain constrained by all original geometry during solving.
 for box,radius,extra in zip(guard.hard_boxes,guard.hard_radii,guard.hard_margins):
  if box[1]<=initial[1] or box[1]-initial[1]>max(speed*horizon,vehicle.half_length_m):continue
  width=float(radius)+float(extra)+body+margin
  if abs(box[0]-initial[0])>width:continue
  offsets[-1]=min(offsets[-1],float(box[0]-initial[0]-width));offsets[1]=max(offsets[1],float(box[0]-initial[0]+width));nearest=min(nearest,float(box[1]-initial[1]-radius-vehicle.half_length_m))
 duration=min(horizon,max(3*interval,3.));rows=[];times=np.arange(7)*interval;rate=np.deg2rad(vehicle.max_steer_rate_degrees);segment=duration/3.
 for sign in (-1,1):
  displacement=offsets[sign];amplitude=displacement*vehicle.wheelbase_m/max(speed*speed*segment*segment,1.)
  # Triangular steering-angle lobes survive coarse knot sampling. A quintic
  # curve's second derivative can be zero at every coarse knot and erase a seed.
  amplitude=float(np.clip(amplitude,-rate*segment/2,rate*segment/2));delta=np.interp(times,[0.,segment,2*segment,3*segment],[0.,amplitude,-amplitude,0.]);delta[0]=initial[4];controls=np.zeros((6,2));controls[:,1]=np.clip(np.diff(delta)/interval,-rate,rate);rows.append(('lane_change_left' if sign<0 else 'lane_change_right',controls))
 return rows,dict(policy='current_geometry_cruising_detour_starts_v1',lateral_offsets_m=[offsets[-1],offsets[1]],duration_s=duration,acceleration_seed_mps2=0.,final_trajectory_not_prescribed=True)
