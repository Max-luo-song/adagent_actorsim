"""Host representation adapter; frozen certificate and candidate policy unchanged."""
import numpy as np
from shapely.geometry import shape,LineString,Point
from scipy.ndimage import distance_transform_edt

def install(frozen):
 original_geometry=frozen.StaticGeometry;original_reference=frozen.MapRoadReference
 class SceneGeometry(original_geometry):
  def __init__(self,mapping,vehicle):
   if 'geometry' not in mapping:return super().__init__(mapping,vehicle)
   self.data=mapping;self.v=vehicle;self.poly=shape(mapping['geometry'])
   if not self.poly.is_valid or self.poly.area<=0 or mapping.get('online_static_map_allowed') is not True or mapping.get('unknown_is_drivable') is not False:raise ValueError('Invalid scene-specific observed road domain')
   self.uncertainty=float(mapping['uncertainty_m']);self.bounds=self.poly.bounds;self.info=None;self.step=.25;x0,y0,x1,y1=self.bounds
   self.xs=np.arange(x0-.5,x1+.51,self.step);self.ys=np.arange(y0-.5,y1+.51,self.step);X,Y=np.meshgrid(self.xs,self.ys)
   try:
    from shapely import contains_xy
    inside=contains_xy(self.poly,X,Y)
   except ImportError:
    from shapely.vectorized import contains
    inside=contains(self.poly,X,Y)
   self.sdf=(distance_transform_edt(inside)-distance_transform_edt(~inside))*self.step
 class SceneReference(original_reference):
  def __init__(self,vehicle,mapping):
   # Retain the frozen superclass initialization and numerical geometry methods.
   old=frozen.StaticGeometry;frozen.StaticGeometry=SceneGeometry
   try:super().__init__(vehicle,mapping)
   finally:frozen.StaticGeometry=old
   self.generic_domain='geometry' in mapping
  def observed(self,obs,info,arrays):
   if not self.generic_domain:return super().observed(obs,info,arrays)
   self.mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'],float);self.geometry.set_pose(info);self.road=self.geometry;self.soft=self.geometry
   origin=-self.mount[:2,3];stations=np.linspace(origin[1]-3,origin[1]+15,37);samples=[];center=origin[0]
   for y in stations:
    ends=self.geometry.world(np.array([[-80.,y],[80.,y]]));cut=self.geometry.poly.intersection(LineString(ends))
    pieces=list(cut.geoms) if hasattr(cut,'geoms') else [cut];ranges=[]
    for part in pieces:
     if part.geom_type!='LineString' or part.length<.1:continue
     p=self.geometry.model(np.asarray(part.coords));ranges.append((float(p[:,0].min()),float(p[:,0].max())))
    if not ranges:continue
    lo,hi=min(ranges,key=lambda r:max(r[0]-center,center-r[1],0.));center=float(np.clip(center,lo,hi));samples.append((y,lo,hi))
   edges=[]
   if len(samples)>=12:
    a=np.asarray(samples)
    for side,index in (('left',1),('right',2)):
     coeff=np.polyfit(a[:,0],a[:,index],1);residual=np.max(np.abs(np.polyval(coeff,a[:,0])-a[:,index]))
     # A poorly represented curve never becomes an extrapolated straight road.
     if residual>self.geometry.uncertainty:continue
     edges.append({'side':side,'a':float(coeff[0]),'b':float(coeff[1]),'valid_y_range':[float(a[0,0]),float(a[-1,0])],'observed_y_range':[float(a[0,0]),float(a[-1,0])],'uncertainty_m':self.geometry.uncertainty})
   self.packet={'schema':'authorized_static_map_local_boundaries_v1','status':'fresh' if len(edges)==2 else 'unavailable','source':'scene_specific_observed_road_domain','edges':edges,'online_static_map_authorized':True,'per_frame_image_detection':False,'unknown_is_drivable':False}
   return self.packet
  def coverage(self,states):
   if not self.generic_domain:return super().coverage(states)
   from shapely.geometry import Polygon
   return all(self.geometry.poly.covers(Polygon(p)) for p in self.geometry.bodies(states))
 frozen.MapRoadReference=SceneReference
 return SceneGeometry,SceneReference
