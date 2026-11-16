"""Transform retained prediction queries; retain current obstacle evidence."""
import numpy as np
from empty_native_temporal import TemporalOccupancy as OriginalTemporal
from mpc_native_v1.native import occupancy_clearance
class TemporalOccupancy:
 def __init__(self,masks,vehicle,info=None,retained=None,**kwargs):
  self.retained=retained;self.vehicle=vehicle;self.age=0.;self.extra=0.;self.matrix=np.eye(2);self.translation=np.zeros(2)
  if retained:
   current=np.asarray(info['explicit_l2g_r_mat']);source=np.asarray(retained['source_rotation']);a=current.T@source;b=(np.asarray(info['explicit_l2g_t'])-np.asarray(retained['source_translation']))@source
   if not np.allclose(a[2],[0,0,1],atol=1e-5):raise ValueError('Nonplanar retained forecast')
   self.matrix=a[:2,:2];self.translation=b[:2];self.age=float(retained['age_s']);self.extra=self.age*float(retained['unobserved_motion_budget_mps'])
   if not np.isfinite(self.extra) or self.extra<0:raise ValueError('Invalid forecast uncertainty budget')
  self.current=FreshGeometry(vehicle,**kwargs) if retained else None
  self.inner=OriginalTemporal(masks,vehicle) if retained else OriginalTemporal(masks,vehicle,**kwargs)
 def __getattr__(self,name):return getattr(self.inner,name)
 def clearance(self,states,times):
  rows=np.array(states,copy=True)
  if self.retained:
   rows[:,:2]=rows[:,:2]@self.matrix+self.translation;direction=np.c_[np.sin(rows[:,2]),np.cos(rows[:,2])]@self.matrix;rows[:,2]=np.arctan2(direction[:,0],direction[:,1])
  result=self.inner.clearance(rows,np.asarray(times)+self.age)-self.extra
  if self.current is not None:result=np.minimum(result,self.current.clearance(states))
  return result
 def audit(self):
  result=self.inner.audit();result['forecast_context']={'source':'retained_model_forecast' if self.retained else 'fresh_model_forecast','age_s':self.age,'uncertainty_expansion_m':self.extra,'current_raster_fabricated':False,'current_obstacle_geometry_not_aged_or_inflated':True,'source_horizon_policy':'existing last forecast bin hold retained explicitly; no new neural prediction'};return result
class ObservedGeometry:
 """Available geometry only, not a declaration of free/unobserved space."""
 def __init__(self,report,vehicle):
  self.report=report;self.vehicle=vehicle
  for name in ['observed_depth_obstacles_xy','unknown_visibility_obstacles_xy']:
   if not np.isfinite(np.asarray(report.get(name,[]),dtype=float).reshape(-1,2)).all():raise ValueError('Invalid observed fallback geometry')
 def clearance(self,states,times):
  result=np.full(len(states),np.inf)
  for name,key in [('observed_depth_obstacles_xy','observed_depth_geometry'),('unknown_visibility_obstacles_xy','unknown_visibility_geometry')]:
   points=np.asarray(self.report.get(name,[]),dtype=float).reshape(-1,2)
   if len(points):result=np.minimum(result,occupancy_clearance(states,points,self.vehicle,max(np.sqrt(3)*.25,float(self.report.get(key,{}).get('point_radius_m',0.)))))
  return result
 def audit(self):return {'raster_available':False,'available_geometry_only':True,'unknown_space_asserted_free':False}

class AgentlessForecast:
 """A structural empty model-agent set, never an empty physical scene/raster."""
 pass

class FreshGeometry:
 def __init__(self,vehicle,**kwargs):
  from empty_bounded_geometry import projected_cell
  from mpc_reuse_v1.cached_geometry import CachedPolygon
  from scipy.spatial import cKDTree
  self.vehicle=vehicle;self.rows=[]
  for prefix in ['observed','unknown']:
   points=np.asarray(kwargs.get(prefix+'_points') if kwargs.get(prefix+'_points') is not None else [],dtype=float).reshape(-1,2);radius=float(kwargs.get(prefix+'_point_radius',np.sqrt(2)*.25));rotation=kwargs.get(prefix+'_cell_rotation')
   if not np.isfinite(points).all() or not np.isfinite(radius) or radius<0:raise ValueError('Invalid current geometry')
   if rotation is not None:
    extra=radius-np.sqrt(3)*.25
    if extra< -1e-10:raise ValueError('Invalid projected voxel uncertainty')
    polygon=projected_cell(rotation);cache=CachedPolygon(points,polygon,vehicle.clearance_margin_m+max(extra,0.)+1e-7,tree=cKDTree(points) if len(points) else None);self.rows.append(('polygon',cache))
   else:self.rows.append(('circle',(points,radius)))
 def clearance(self,states):
  result=np.full(len(states),np.inf);expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
  for kind,data in self.rows:
   if kind=='polygon':result=np.minimum(result,data.clearance(expanded))
   elif len(data[0]):result=np.minimum(result,occupancy_clearance(states,data[0],self.vehicle,data[1]))
  return result

class AgentlessTemporal:
 def __init__(self,vehicle,**kwargs):self.vehicle=vehicle;self.geometry=FreshGeometry(vehicle,**kwargs);self.retained=None;self.extra=0.
 def clearance(self,states,times):
  states=np.asarray(states);times=np.asarray(times)
  if len(states)!=len(times) or not np.isfinite(times).all() or np.any(times<0):raise ValueError('Invalid foreground query clock')
  c=np.abs(np.cos(states[:,2]));s=np.abs(np.sin(states[:,2]));hx=self.vehicle.half_width_m*c+self.vehicle.half_length_m*s;hy=self.vehicle.half_width_m*s+self.vehicle.half_length_m*c;coverage=np.minimum(50.-np.abs(states[:,0])-hx,50.-np.abs(states[:,1])-hy)-self.vehicle.clearance_margin_m
  return np.minimum(coverage,self.geometry.clearance(states))
 def audit(self):return {'source':'healthy_no_modelled_foreground_queries','foreground_query_count':0,'raster_available':False,'raster_fabricated':False,'physical_scene_asserted_empty':False,'current_static_and_unknown_constraints_preserved':True,'future_unmodelled_agents_not_certified':True}
