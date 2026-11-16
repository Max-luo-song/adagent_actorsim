"""RGB-D visible-face fitting; no asset geometry, scene IDs or training."""
import numpy as np
from depth_contract import reliable_mask,geometry_depth
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

def points_from_depth(obs,info,stride=4):
 from input_geometry import intrinsic_matrix
 all_points=[]
 for name in obs['depth']:
  depth=geometry_depth(obs,name)
  sample_stride=4*int(np.ceil(stride/4)) if name in obs.get('surface_depth',{}) else stride
  k=intrinsic_matrix(info['cam_params'][name]['intrinsic'])[:3,:3];h,w=depth.shape;yy,xx=np.mgrid[0:h:sample_stride,0:w:sample_stride];z=depth[yy,xx];cam=np.stack([(xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z],-1).reshape(-1,3);inv=np.linalg.inv(info['cam_params'][name]['l2c']);points=(inv@np.c_[cam,np.ones(len(cam))].T).T[:,:3];valid=np.isfinite(points).all(1)&(cam[:,2]>.1)&(np.linalg.norm(points[:,:2],axis=1)<50)
  valid &= reliable_mask(obs,name)[yy,xx].reshape(-1)
  all_points.append(points[valid])
 return np.concatenate(all_points)

def ground_plane(obs,info):
 points=points_from_depth(obs,info,6);mount=np.array(info['model_lidar_frame']['virtual_to_ego']);z0=-mount[2,3];points=points[(abs(points[:,2]-z0)<.75)&(np.linalg.norm(points[:,:2],axis=1)<30)]
 if len(points)<100:return None
 rng=np.random.default_rng(0)
 if len(points)>3000:points=points[rng.choice(len(points),3000,replace=False)]
 design=np.c_[points[:,:2],np.ones(len(points))];best=None
 for _ in range(100):
  sample=rng.choice(len(points),3,replace=False)
  try:plane=np.linalg.solve(design[sample],points[sample,2])
  except np.linalg.LinAlgError:continue
  if np.linalg.norm(plane[:2])>np.tan(np.deg2rad(15)):continue
  residual=abs(design@plane-points[:,2]);inside=residual<.12
  if best is None or inside.sum()>best.sum():best=inside
 if best is None or best.sum()<100:return None
 return np.linalg.lstsq(design[best],points[best,2],rcond=None)[0]


def projected_box_roi(camera_corners,k,width,height,near=.1):
 """Clip the actual box edges at the camera near plane before projection.

 A partially visible box must not be discarded just because one corner is
 behind the camera. Fully hidden boxes still produce no observation.
 """
 cp=np.asarray(camera_corners,dtype=float)
 if cp.shape!=(8,3) or not np.isfinite(cp).all() or near<=0:raise ValueError('Invalid camera box')
 edges=((0,4),(4,7),(7,3),(3,0),(1,5),(5,6),(6,2),(2,1),(0,1),(4,5),(7,6),(3,2))
 points=[p for p in cp if p[2]>=near]
 for a,b in edges:
  if (cp[a,2]<near)!=(cp[b,2]<near):
   t=(near-cp[a,2])/(cp[b,2]-cp[a,2]);points.append(cp[a]+t*(cp[b]-cp[a]))
 if not points:return None
 points=np.asarray(points);uv=points@np.asarray(k).T;uv=uv[:,:2]/points[:,2,None]
 extent=np.ptp(uv,axis=0);lo=np.maximum(np.floor(uv.min(0)-.15*extent),0);hi=np.minimum(np.ceil(uv.max(0)+.15*extent)+1,[width,height])
 if np.any(hi<=lo):return None
 return lo.astype(int),hi.astype(int)

def extract_surface(obs,info,box,plane):
 from input_geometry import intrinsic_matrix
 from box_geometry import box_corners
 points=[];origins=[]
 for name in obs['depth']:
  depth=geometry_depth(obs,name)
  k=intrinsic_matrix(info['cam_params'][name]['intrinsic'])[:3,:3];l2c=np.array(info['cam_params'][name]['l2c']);corners=box_corners(box);cp=(l2c@np.c_[corners,np.ones(8)].T).T[:,:3]
  roi=projected_box_roi(cp,k,depth.shape[1],depth.shape[0])
  if roi is None:continue
  lo,hi=roi
  step=4 if name in obs.get('surface_depth',{}) else 2
  lo=(lo+step-1)//step*step
  yy,xx=np.mgrid[lo[1]:hi[1]:step,lo[0]:hi[0]:step];z=depth[yy,xx];cam=np.stack([(xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z],-1).reshape(-1,3);inv=np.linalg.inv(l2c);p=(inv@np.c_[cam,np.ones(len(cam))].T).T[:,:3];height=p[:,2]-(np.c_[p[:,:2],np.ones(len(p))]@plane);keep=np.isfinite(p).all(1)&(cam[:,2]>.1)&(height>.2)&(height<max(box[5]*1.5,box[5]+.5))&(np.linalg.norm(p[:,:2]-box[:2],axis=1)<np.hypot(*box[3:5])/2+2.)
  keep &= reliable_mask(obs,name)[yy,xx].reshape(-1)
  p=p[keep];points.extend(p);origins.extend(np.repeat(inv[None,:2,3],len(p),axis=0))
 if len(points)<50:return None
 points=np.array(points);origins=np.array(origins);_,indices=np.unique(np.floor(points/.08).astype(int),axis=0,return_index=True);points=points[indices];origins=origins[indices];labels=DBSCAN(eps=.24,min_samples=5).fit_predict(points);clusters=[np.flatnonzero(labels==i) for i in set(labels) if i>=0]
 if not clusters:return None
 cluster=min(clusters,key=lambda indices:np.linalg.norm(np.median(points[indices,:2],axis=0)-box[:2]));p=points[cluster];o=origins[cluster]
 if len(p)<50:return None
 d=p[:,:2]-o;r=np.linalg.norm(d,axis=1);angles=np.arctan2(d[:,0],d[:,1]);bins=np.floor(angles/.005).astype(int);contour=[]
 for key in np.unique(bins):
  idx=np.flatnonzero(bins==key);limit=np.quantile(r[idx],.05)+.08;near=idx[r[idx]<=limit];contour.append(int(near[np.argmin(r[near])]))
 if len(contour)<8:return None
 return p,o,p[contour,:2],o[contour]

def axes(yaw):return np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])

def refine(obs,info,box,plane,fit_yaw=False):
 original=np.array(box,dtype=float);surface=extract_surface(obs,info,original,plane)
 if surface is None:return dict(accepted=False,reason='insufficient supported surface',box=original.tolist())
 points,origins,contour,cam=surface;radius=np.linalg.norm(contour-cam,axis=1);directions=(contour-cam)/radius[:,None];half=original[3:5]/2
 def residual(x,prior=True):
  basis=axes(x[2]);origin=(cam-x[:2])@basis;direction=directions@basis;direction=np.where(abs(direction)<1e-8,np.sign(direction+1e-12)*1e-8,direction);a=(-half-origin)/direction;b=(half-origin)/direction;entry=np.minimum(a,b).max(1);exit=np.maximum(a,b).min(1);outside=abs((points[:,:2]-x[:2])@basis)-half;dist=np.linalg.norm(np.maximum(outside,0),axis=1)
  out=np.r_[np.clip(entry-radius,-5,5),np.clip(entry-exit,0,5),dist*.5]
  if prior:out=np.r_[out,.05*(x[:2]-original[:2]),.05*np.sin(x[2]-original[6])]
  return out
 lower=np.r_[original[:2]-2.,original[6]-np.pi/2];upper=np.r_[original[:2]+2.,original[6]+np.pi/2];seeds=[np.r_[original[:2],original[6]]]
 for angle in ([0.,-.785398,.785398,-1.4,1.4] if fit_yaw else [0.]):
  yaw=original[6]+angle;basis=axes(yaw);lo,hi=np.quantile(points[:,:2]@basis,[.005,.995],axis=0);camera=np.median(origins,axis=0)@basis;mid=(lo+hi)/2;coverage=hi-lo
  for k in range(2):
   if coverage[k]<.7*original[3+k]:mid[k]=lo[k]+half[k] if camera[k]<mid[k] else hi[k]-half[k]
  center=basis@mid;seeds.append(np.clip(np.r_[center,yaw],lower+1e-8,upper-1e-8))
 best=None
 for seed in seeds:
  if fit_yaw:
   fit=least_squares(residual,seed,bounds=(lower,upper),loss='soft_l1',f_scale=.15,max_nfev=180)
  else:
   fit=least_squares(lambda xy:residual(np.r_[xy,original[6]]),seed[:2],bounds=(lower[:2],upper[:2]),loss='soft_l1',f_scale=.15,max_nfev=180)
   fit.x=np.r_[fit.x,original[6]]
  if best is None or fit.cost<best.cost:best=fit
 before=float(np.mean(abs(residual(seeds[0],False))));after=float(np.mean(abs(residual(best.x,False))));new=original.copy();new[:2]=best.x[:2];new[6]=best.x[2];accepted=bool(after<before-.02 and np.linalg.norm(new[:2]-original[:2])<1.95)
 return dict(accepted=accepted,reason='visible-face residual improved' if accepted else 'no reliable improvement',box=(new if accepted else original).tolist(),proposed_box=new.tolist(),before_surface_residual_m=before,after_surface_residual_m=after,center_change_m=float(np.linalg.norm(new[:2]-original[:2])),yaw_change_degrees=float(np.degrees(np.arctan2(np.sin(new[6]-original[6]),np.cos(new[6]-original[6])))),surface_points=len(points),contour_points=len(contour),amodal_dimensions_source='unchanged neural prediction',yaw_refitted=fit_yaw,full_geometry_certified=False,points=points,contour=contour)
