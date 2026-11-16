"""Sensor-only ground-height intervals; no remote single-plane extrapolation.

The ground consensus, 15 degree grade and 0.12m fit tolerance are inherited
from depth_surface_geometry.ground_plane. They are model assumptions, not a
certified terrain bound. Point and reference depth error are propagated
separately; a Gaussian scale/opacity test is never a height-error certificate.
"""
import numpy as np
from scipy.spatial import cKDTree
from depth_surface_geometry import points_from_depth
GRADE=np.tan(np.deg2rad(15.))
FIT_TOLERANCE=.12
class GroundReference:
 def __init__(self,points,point_error):
  self.points=np.asarray(points,dtype=float).reshape(-1,3)
  if not np.isfinite(self.points).all() or not np.isfinite(point_error) or point_error<0:raise ValueError('Invalid ground reference observations/error')
  self.error=float(point_error)+FIT_TOLERANCE
  self.tree=cKDTree(self.points[:,:2]) if len(self.points) else None
 def bounds(self,xy):
  xy=np.asarray(xy,dtype=float).reshape(-1,2)
  if self.tree is None:return np.full(len(xy),-np.inf),np.full(len(xy),np.inf)
  # Any subset gives an enclosing interval; nearest samples tighten it locally.
  distance,index=self.tree.query(xy,k=min(32,len(self.points)))
  if distance.ndim==1:distance=distance[:,None];index=index[:,None]
  z=self.points[index,2];delta=self.error+GRADE*distance
  return np.max(z-delta,axis=1),np.min(z+delta,axis=1)
 def audit(self):return dict(policy='sensor_ground_grade_interval_v1',ground_samples=len(self.points),maximum_grade_degrees=15.,reference_depth_and_fit_error_m=self.error,extrapolation='interval grows with distance to observed ground support',certified=False)

def from_observation(obs,info,plane,point_error):
 if plane is None:return GroundReference([],point_error)
 points=points_from_depth(obs,info,6);mount=np.asarray(info['model_lidar_frame']['virtual_to_ego']);z0=-mount[2,3]
 points=points[(abs(points[:,2]-z0)<.75)&(np.linalg.norm(points[:,:2],axis=1)<30.)]
 residual=abs(np.c_[points[:,:2],np.ones(len(points))]@plane-points[:,2]);points=points[residual<FIT_TOLERANCE]
 if len(points)<100:points=np.empty((0,3))
 return GroundReference(points,point_error)
