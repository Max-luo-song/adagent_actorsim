"""Same bounded square-cell SAT and pruning, using the verified native kernel."""
import numpy as np
from mpc_native_v1.native import rectangle_separation
class NativeRoadClearance:
 def __init__(self,road):self.road=road
 def __call__(self,states):
  states=np.asarray(states,dtype=float)
  if states.ndim!=2 or states.shape[1]!=5 or not np.isfinite(states).all():raise ValueError('Invalid drivable footprint query')
  if len(states)>32:return np.concatenate([self(states[i:i+32]) for i in range(0,len(states),32)])
  road=self.road;c=abs(np.cos(states[:,2]));s=abs(np.sin(states[:,2]));hx=road.vehicle.half_width_m*c+road.vehicle.half_length_m*s;hy=road.vehicle.half_width_m*s+road.vehicle.half_length_m*c;x0,y0,x1,y1=road.bounds
  result=np.minimum.reduce([states[:,0]-hx-x0,x1-states[:,0]-hx,states[:,1]-hy-y0,y1-states[:,1]-hy])-road.vehicle.clearance_margin_m
  if road.tree is not None and len(states):
   distance,nearest=road.tree.query(states[:,:2]);near=road.cell_boxes[nearest];upper=np.diag(rectangle_separation(states,near,np.zeros(len(states)),road.vehicle.half_width_m,road.vehicle.half_length_m));body=np.hypot(road.vehicle.half_width_m,road.vehicle.half_length_m);radius=np.sqrt(2.)*(upper+body+road.radius)+1e-8;neighbors=road.tree.query_ball_point(states[:,:2],np.maximum(radius,distance+1e-8));ids=np.unique(np.concatenate(neighbors)).astype(int);gaps=rectangle_separation(states,road.cell_boxes[ids],np.zeros(len(states)),road.vehicle.half_width_m,road.vehicle.half_length_m);result=np.minimum(result,gaps.min(axis=1)-road.vehicle.clearance_margin_m)
  return result
