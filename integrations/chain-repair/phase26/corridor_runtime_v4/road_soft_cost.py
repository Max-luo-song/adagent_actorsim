"""A directional soft cost from the current model road mask; never a hard gate."""
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
class SoftRoadPenalty:
 def __init__(self,road,vehicle):
  self.road=road;self.vehicle=vehicle
  self.has_road=bool(np.any(road.mask))
  self.x0,self.y0,self.x1,self.y1=map(float,road.bounds)
  self.dx=float(road.dx);self.dy=float(road.dy)
  self.coefficients=np.array([(x,y) for x in (-1.,0.,1.) for y in (-1.,0.,1.)])
  if self.has_road:
   mask=np.pad(np.asarray(road.mask,dtype=bool),1,constant_values=False)
   self.sdf=(distance_transform_edt(mask,sampling=(self.dy,self.dx))-distance_transform_edt(~mask,sampling=(self.dy,self.dx)))[1:-1,1:-1]
  else:self.sdf=None
 def _violations(self,states,stride=1):
  states=np.asarray(states,dtype=float)
  if states.ndim!=2 or states.shape[1]!=5 or not np.isfinite(states).all() or stride<=0:raise ValueError('Invalid soft road query')
  if not self.has_road:return np.zeros((len(states),9))  # No invented road.
  c=np.cos(states[:,2]);s=np.sin(states[:,2]);right=np.c_[c,-s];forward=np.c_[s,c]
  points=states[:,None,:2]+self.coefficients[None,:,0,None]*self.vehicle.half_width_m*right[:,None,:]+self.coefficients[None,:,1,None]*self.vehicle.half_length_m*forward[:,None,:]
  clipped=points.copy();clipped[:,:,0]=np.clip(clipped[:,:,0],self.x0+self.dx/2,self.x1-self.dx/2);clipped[:,:,1]=np.clip(clipped[:,:,1],self.y0+self.dy/2,self.y1-self.dy/2)
  coords=np.array([(clipped[:,:,1]-self.y0)/self.dy-.5,(clipped[:,:,0]-self.x0)/self.dx-.5])
  values=map_coordinates(self.sdf,coords,order=1,mode='nearest',prefilter=False)
  outside=np.linalg.norm(points-clipped,axis=2)
  violation=np.maximum(self.vehicle.clearance_margin_m-values+outside,0.)
  return violation
 def distance_cost(self,states,stride):
  return float(np.sum(np.mean(self._violations(states,stride)**2,axis=1))/stride)
 def peak_violation(self,states,buffer_m=0.,start_s=0.):
  if not self.has_road:return 0.
  states=np.asarray(states);start=max(0,int(round(start_s/self.vehicle.dt)));query=states[start:]
  if not len(query):return 0.
  violation=self._violations(query);edge=np.maximum(float(buffer_m)-np.asarray(self.road.clearance(query)),0.)
  return float(max(np.max(violation**2),np.max(edge**2)))
 def cost(self,states,stride):
  overlap=float(np.sum(np.minimum(self.road.clearance(states),0.)**2)/stride)
  return overlap+self.distance_cost(states,stride)
 def audit(self):
  return dict(policy='overlap_plus_continuous_model_road_distance_v1',road_cells_present=self.has_road,footprint_samples=9,interpolation='bilinear',hard_constraint=False,road_prediction_certified=False,missing_road_does_not_create_return_direction=True)
