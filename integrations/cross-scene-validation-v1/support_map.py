"""Shared conservative reconstruction-support adapter. Holes remain unknown."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape,Polygon,LineString
from shapely.vectorized import contains

def install(module):
 FrozenGeometry=module.StaticGeometry
 class Geometry(FrozenGeometry):
  def __init__(self,mapping,vehicle):
   self.data=mapping;self.v=vehicle;self.poly=shape(mapping['geometry'])
   if not self.poly.is_valid or self.poly.is_empty:raise ValueError('Invalid support geometry')
   if mapping.get('authority')!='reconstructed_support_proxy' or mapping.get('holes_preserved') is not True:raise ValueError('Undeclared map authority')
   self.uncertainty=float(mapping['uncertainty_m']);self.bounds=self.poly.bounds;self.info=None;self.step=.25
   x0,y0,x1,y1=self.bounds;self.xs=np.arange(x0-.5,x1+.51,self.step);self.ys=np.arange(y0-.5,y1+.51,self.step)
   X,Y=np.meshgrid(self.xs,self.ys);inside=contains(self.poly,X,Y);self.sdf=(distance_transform_edt(inside)-distance_transform_edt(~inside))*self.step
 def observed(self,obs,info,arrays):
  self.mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'],float);self.geometry.set_pose(info);self.road=self.geometry;self.soft=self.geometry
  origin=-self.mount[:2,3];stations=np.linspace(origin[1]-3,origin[1]+15,12);rows=[]
  for y in stations:
   line=LineString(self.geometry.world(np.array([[-500.,y],[500.,y]])));cut=self.geometry.poly.intersection(line)
   pieces=list(cut.geoms) if hasattr(cut,'geoms') else [cut];intervals=[]
   for piece in pieces:
    if piece.geom_type!='LineString' or piece.is_empty:continue
    local=self.geometry.model(np.asarray(piece.coords)[:,:2]);intervals.append((float(local[:,0].min()),float(local[:,0].max())))
   if intervals:
    lo,hi=min(intervals,key=lambda q:max(q[0]-origin[0],origin[0]-q[1],0));rows.append((y,lo,hi))
  edges=[]
  if len(rows)>=2:
   rows=np.asarray(rows)
   for side,col in [('left',1),('right',2)]:
    a,b=np.polyfit(rows[:,0],rows[:,col],1);residual=rows[:,col]-(a*rows[:,0]+b);b+=float(residual.max() if side=='left' else residual.min())
    edges.append(dict(side=side,a=float(a),b=float(b),valid_y_range=[float(rows[0,0]),float(rows[-1,0])],observed_y_range=[float(rows[0,0]),float(rows[-1,0])],uncertainty_m=self.geometry.uncertainty))
  self.packet=dict(schema='reconstruction_support_local_cross_sections_v1',status='fresh' if edges else 'unavailable',source='scene-matched reconstructed road support; not surveyed white line',edges=edges,online_static_map_authorized=True,physical_white_boundary_certified=False)
  return self.packet
 def coverage(self,states):return bool(all(self.geometry.poly.covers(Polygon(p)) for p in self.geometry.bodies(states)))
 module.StaticGeometry=Geometry;module.MapRoadReference.observed=observed;module.MapRoadReference.coverage=coverage
 return module
