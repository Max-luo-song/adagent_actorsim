"""Current RGB solid road-edge evidence; never simulator road truth.

Conservative scope: locally straight, continuous WHITE edge with a visible
pavement/shoulder colour transition. Other markings remain unclassified.
Camera-ground intersection is an approximation, explicitly bounded in range.
"""
import copy
import numpy as np
SCHEMA='observed_white_road_edge_v1'

def geometry(info,name,shape):
 p=info['cam_params'][name]; k=p['intrinsic'];h,w=shape[:2]
 K=np.array([[w/(2*np.tan(k['fovx']/2)),0,k['cx']],[0,h/(2*np.tan(k['fovy']/2)),k['cy']],[0,0,1.]])
 M=np.asarray(p['l2c'],float);z=-float(np.asarray(info['model_lidar_frame']['virtual_to_ego'])[2,3])
 if not np.isfinite(K).all() or M.shape!=(4,4) or not np.isfinite(M).all() or not .5<-z<4:raise ValueError('Invalid calibrated camera/ground plane')
 return K,M,z

def project(xyz,K,M):
 c=np.asarray(xyz)@M[:3,:3].T+M[:3,3];uv=c@K.T
 return uv[:,:2]/np.maximum(uv[:,2,None],1e-10),c[:,2]

def unproject(uv,K,M,z):
 inv=np.linalg.inv(M);ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T@inv[:3,:3].T
 t=(z-inv[2,3])/ray[:,2]
 return inv[:3,3]+ray*t[:,None]

def detect(observation,info):
 import cv2
 timestamp=float(info['timestamp']);packet=dict(schema=SCHEMA,status='unavailable',timestamp=timestamp,source='current_camera_rgb_ground_projection',ground_truth_used=False,edges=[],reason='no_qualified_solid_road_edge',plane_projection_approximate=True)
 if not np.isfinite(timestamp):raise ValueError('Nonfinite edge clock')
 proposals=[]
 # Front-side cameras can supplement a partly out-of-view road edge.
 for name in ('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'):
  rgb=observation.get('rgb',{}).get(name)
  if rgb is None:continue
  rgb=np.asarray(rgb)
  if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Invalid RGB edge input')
  h,w=rgb.shape[:2];K,M,z=geometry(info,name,rgb.shape)
  lum=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);contrast=lum.astype(float)-cv2.medianBlur(lum,25).astype(float)
  white=(rgb.max(2).astype(float)-rgb.min(2)<45)&(lum>110)&(contrast>15)
  white[:int(h*.51)]=False
  lines=cv2.HoughLinesP(white.astype(np.uint8)*255,1,np.pi/720,threshold=45,minLineLength=80,maxLineGap=12)
  if lines is None:continue
  ys=np.arange(6.,45.01,.25);seen=[]
  def sample(x,y):
   uv,d=project(np.c_[x,y,np.full(len(y),z)],K,M);valid=(d>0)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
   pix=np.rint(np.clip(uv,[0,0],[w-1,h-1])).astype(int)
   return pix,valid
  for uv in lines[:,0].reshape(-1,2,2):
   g=unproject(uv,K,M,z);x,y=g[:,:2].T
   if np.any(y<2) or np.any(y>70) or np.any(abs(x)>18) or abs(np.diff(y)[0])<5:continue
   a=np.diff(x)[0]/np.diff(y)[0];b=x[0]-a*y[0]
   if abs(a)>.65 or any(abs(a-aa)<.015 and abs(b-bb)<.2 for aa,bb in seen):continue
   seen.append((a,b))
   # Refine only around an observed Hough segment, not a fabricated lane width.
   for _ in range(2):
    lateral=np.arange(-.20,.201,.025);xx=a*ys[:,None]+b+lateral
    pix,valid=sample(xx.ravel(),np.repeat(ys,len(lateral)));valid=valid.reshape(xx.shape)
    good=white[pix[:,1],pix[:,0]].reshape(xx.shape)&valid
    strength=contrast[pix[:,1],pix[:,0]].reshape(xx.shape);strength=np.where(good,strength,-1)
    peak=strength.argmax(1);hit=good.any(1);xs=xx[np.arange(len(ys)),peak]
    if hit.sum()<40:break
    a,b=np.polyfit(ys[hit],xs[hit],1)
   if hit.sum()<40:continue
   residual=np.abs(xs-(a*ys+b));hit&=residual<.15
   hit_ys=ys[hit]
   if len(hit_ys)<40:continue
   lo,hi=float(hit_ys.min()),float(hit_ys.max());span=hi-lo
   # Require a continuous 12m+ edge; dashed lane separators fail coverage/gaps.
   region=(ys>=lo)&(ys<=hi);coverage=float(hit[region].mean())
   maxgap=float(np.diff(hit_ys).max())
   if span<12 or coverage<.85 or maxgap>1.25:continue
   yq=ys[hit];xq=a*yq+b
   colours=[];valids=[]
   for offset in (-.8,.8):
    pix,valid=sample(xq+offset,yq);colours.append(rgb[pix[:,1],pix[:,0]].astype(float));valids.append(valid)
   valid=valids[0]&valids[1]
   if valid.sum()<40:continue
   left,right=[c[valid] for c in colours]
   # Distinguish a shoulder from a solid divider within similar pavement.
   transition=float(np.median((right[:,0]-right[:,2])-(left[:,0]-left[:,2]))/255)
   if abs(transition)<.025:continue
   side='right' if transition>0 else 'left'
   noise=float(np.quantile(residual[hit],.95));near=max(-5.,lo-12.);far=hi+3.
   proposals.append(dict(side=side,a=float(a),b=float(b),observed_y_range=[lo,hi],valid_y_range=[near,far],uncertainty_m=max(.15,noise*2),camera=name,coverage=coverage,max_gap_m=maxgap,shoulder_transition=transition,support_points=int(hit.sum()),plane_height_m=z))
 # Choose strongest continuous edge per side, not a fixed ego/lane width.
 for side in ('left','right'):
  candidates=[p for p in proposals if p['side']==side]
  if candidates:packet['edges'].append(max(candidates,key=lambda p:p['coverage']*(p['observed_y_range'][1]-p['observed_y_range'][0])))
 if packet['edges']:packet.update(status='fresh',reason=None)
 return packet

class WhiteEdge:
 def __init__(self,packet,vehicle):
  self.vehicle=vehicle;self.packet=copy.deepcopy(packet);self.edges=[]
  if packet is None:return
  if packet.get('schema')!=SCHEMA or packet.get('ground_truth_used') is not False or packet.get('status') not in ('fresh','unavailable'):raise ValueError('Invalid white edge provenance/status')
  for e in packet.get('edges',[]):
   nums=[e['a'],e['b'],*e['valid_y_range'],*e['observed_y_range'],e['uncertainty_m']]
   if not np.isfinite(nums).all() or e['side'] not in ('left','right') or e['valid_y_range'][1]<=e['valid_y_range'][0] or e['uncertainty_m']<.15:raise ValueError('Invalid white edge geometry')
   self.edges.append(e)
  if packet['status']=='fresh' and not self.edges:raise ValueError('Fresh edge without geometry')
 def clearance(self,states):
  s=np.asarray(states,float)
  if s.ndim!=2 or s.shape[1]<3 or not np.isfinite(s).all():raise ValueError('Invalid edge states')
  result=np.full(len(s),np.inf);v=self.vehicle
  local=np.array([[-v.half_width_m,-v.half_length_m],[v.half_width_m,-v.half_length_m],[v.half_width_m,v.half_length_m],[-v.half_width_m,v.half_length_m]])
  c=np.cos(s[:,2]);t=np.sin(s[:,2]);x=s[:,0,None]+c[:,None]*local[:,0]+t[:,None]*local[:,1];y=s[:,1,None]-t[:,None]*local[:,0]+c[:,None]*local[:,1]
  for e in self.edges:
   a,b=e['a'],e['b'];lo,hi=e['valid_y_range'];direction=1. if e['side']=='right' else -1.
   # Clip each complete body rectangle to the longitudinal observation band.
   xx=[x];yy=[y];vv=[(y>=lo)&(y<=hi)]
   for end in (lo,hi):
    xn=np.roll(x,-1,axis=1);yn=np.roll(y,-1,axis=1);dy=yn-y
    u=(end-y)/np.where(abs(dy)>1e-10,dy,1.);valid=(abs(dy)>1e-10)&(u>=0)&(u<=1)
    xx.append(x+u*(xn-x));yy.append(np.full_like(y,end));vv.append(valid)
   X=np.concatenate(xx,1);Y=np.concatenate(yy,1);valid=np.concatenate(vv,1)
   extrap=np.maximum(e['observed_y_range'][0]-Y,0)+np.maximum(Y-e['observed_y_range'][1],0)
   gap=direction*(a*Y+b-X)/np.sqrt(1+a*a)-v.clearance_margin_m-e['uncertainty_m']-.02*extrap
   result=np.minimum(result,np.where(valid,gap,np.inf).min(1))
  return result
 def swept_clearance(self,states):
  from planning_guard import sweep_allowance
  return self.clearance(states)-sweep_allowance(np.asarray(states),self.vehicle)
 def crop(self,mask,bounds):
  mask=np.asarray(mask,bool).copy();x0,y0,x1,y1=bounds;h,w=mask.shape
  x=x0+(np.arange(w)+.5)*(x1-x0)/w;y=y0+(np.arange(h)+.5)*(y1-y0)/h
  X,Y=np.meshgrid(x,y)
  for e in self.edges:
   lo,hi=e['valid_y_range'];cell_support=.5*(x1-x0)/w+abs(e['a'])*.5*(y1-y0)/h
   inside=(e['a']*Y+e['b']-X)*(1 if e['side']=='right' else -1)>=e['uncertainty_m']*np.sqrt(1+e['a']**2)+cell_support
   mask&=inside|((Y<lo)|(Y>hi))
  return mask

def restrict_road(contract,packet,vehicle):
 """Derived mask AFTER temporal fusion; preserve original neural arrays."""
 from drivable_area import DrivableArea
 result=copy.deepcopy(contract);raw=DrivableArea(contract,vehicle).mask
 result['model_temporal_drivable_mask']=copy.deepcopy(contract.get('temporal_drivable_mask',raw.tolist()))
 result['temporal_drivable_mask']=WhiteEdge(packet,vehicle).crop(raw,contract['bounds_xy']).tolist()
 result['road_temporal_consensus']=dict(result['road_temporal_consensus'],post_consensus_restriction=SCHEMA)
 # Preserve both original model and fused model arrays separately from clipping.
 result['observed_white_edge']=copy.deepcopy(packet)
 result['white_edge_clipped_cells']=int(np.count_nonzero(raw&~np.asarray(result['temporal_drivable_mask'])))
 return result
