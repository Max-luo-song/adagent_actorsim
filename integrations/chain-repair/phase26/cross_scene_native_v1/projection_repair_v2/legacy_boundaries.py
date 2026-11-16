"""Offline multi-view sensor map. Unknown cells and occluded votes stay unknown."""
import numpy as np
from PIL import Image
from shapely.geometry import box,mapping,shape,Polygon
from shapely.ops import unary_union
from scene_contracts import *
PARAMS=dict(cell_m=.5,frame_stride=5,max_local_range_m=40.,min_independent_frames=2,min_road_fraction=.8,uncertainty_m=.3,semantic_confidence_u8=128,semantic_margin_u8=26,occlusion_tolerance_m=.5,max_plane_residual_m=.15)

def fit_ground(lidar):
 p=np.asarray(lidar);p=p[(p[:,10]==1)&(np.abs(p[:,4])<25)&(p[:,3]>-10)&(p[:,3]<45),3:6]
 if len(p)<100:raise ValueError('Insufficient observed geometric ground')
 A=np.c_[p[:,:2],np.ones(len(p))];good=np.ones(len(p),bool)
 for _ in range(4):
  fit=np.linalg.lstsq(A[good],p[good,2],rcond=None)[0];err=np.abs(A@fit-p[:,2]);good=err<=max(.08,float(np.quantile(err,.7)))
 residual=float(np.quantile(err[good],.95))
 if residual>PARAMS['max_plane_residual_m']:raise ValueError('Ground is not locally planar enough for projection')
 return fit,residual

def polygon_from_mask(mask,origin,step):
 rows=[]
 for j,row in enumerate(np.asarray(mask,bool)):
  edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
  for start,end in edges.reshape(-1,2):rows.append(box(origin[0]+(start-.5)*step,origin[1]+(j-.5)*step,origin[0]+(end-.5)*step,origin[1]+(j+.5)*step))
 if not rows:raise ValueError('No multi-frame road support; no guessed width')
 return unary_union(rows)

def camera_projection(points,camera_pose,calibration,shape_wh):
 T=np.eye(4);T[:3,:3]=calibration['opencv_to_waymo'];T=np.asarray(camera_pose)@np.asarray(calibration['camera_waymo_to_ego'])@T
 inv=np.linalg.inv(T);p=np.asarray(points)@inv[:3,:3].T+inv[:3,3]
 fx,fy,cx,cy,*dist=calibration['intrinsics_9'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
 safe=np.where(np.abs(p[:,2])>1e-12,p[:,2],np.nan)
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  x=p[:,0]/safe;y=p[:,1]/safe;r2=x*x+y*y;k1,k2,p1,p2,k3=dist;radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
  uv=np.c_[fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+cx,fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+cy]
 w,h=shape_wh
 valid=(p[:,2]>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
 pixels=np.rint(np.clip(np.nan_to_num(uv),[0,0],[w-1,h-1])).astype(int)
 return pixels,p[:,2],valid

def build(scene):
 from scipy.ndimage import minimum_filter
 bundle=ASSETS/str(scene);source=read(bundle/'source_manifest.json');raw=read(bundle/'raw_metadata/raw_metadata.json');sem=read(bundle/'semantic19/semantic_manifest.json')
 if source['scene']!=scene or raw['scene_index']!=scene:raise ValueError('Wrong scene assets')
 if not raw['raw_processed_pose_identity_all_frames'] or not raw['raw_processed_calibration_identity']:raise ValueError('Unverified raw calibration')
 folder=OUT/'maps'/str(scene);folder.mkdir(parents=True,exist_ok=True);file=folder/'map.json'
 if file.exists():
  result=read(file)
  if result['parameters']!=PARAMS:raise ValueError('Map construction parameters changed; freeze a new map release')
  for entry in result['sources']:verify(entry)
  verify(result['support']);return result
 poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];origin_pose=np.loadtxt(Path(source['raw_data'])/'ego_pose/000.txt');world_from_raw=np.linalg.inv(origin_pose)
 step=PARAMS['cell_m'];low=np.floor((poses[:,:2,3].min(0)-45)/step)*step;high=np.ceil((poses[:,:2,3].max(0)+45)/step)*step
 xs=np.arange(low[0],high[0]+step/2,step);ys=np.arange(low[1],high[1]+step/2,step);X,Y=np.meshgrid(xs,ys);road=np.zeros(X.shape,np.uint16);seen=np.zeros_like(road)
 catalog={(r['frame'],r['camera']):r for r in sem['maps']};sources=[asset(bundle/'source_manifest.json'),asset(bundle/'raw_metadata/raw_metadata.json'),asset(bundle/'semantic19/semantic_manifest.json'),asset(bundle/'poses_and_clock.npz'),asset(Path(source['raw_data'])/'ego_pose/000.txt')];receipts=[]
 frames=list(range(0,source['frames'],PARAMS['frame_stride']))
 if frames[-1]!=source['frames']-1:frames.append(source['frames']-1)
 for frame in frames:
  write(folder/'progress.json',{'stage':'projecting_original_sensors','scene':scene,'frame':frame,'total_source_frames':source['frames']})
  record=source['frames_data'][frame];lidar_asset=record['lidar'];verify(lidar_asset);sources.append(lidar_asset);lidar=np.memmap(lidar_asset['path'],dtype='<f4',mode='r').reshape(-1,14)
  plane,residual=fit_ground(lidar);pose=poses[frame];normal=pose[:3,:3]@np.r_[-plane[:2],1.];offset=plane[2]+normal@pose[:3,3]
  if abs(normal[2])<.5:raise ValueError('Unsupported near-vertical ground')
  Z=(offset-normal[0]*X-normal[1]*Y)/normal[2];world=np.c_[X.ravel(),Y.ravel(),Z.ravel()];local=(world-pose[:3,3])@pose[:3,:3];near=np.linalg.norm(local[:,:2],axis=1)<=PARAMS['max_local_range_m'];ids=np.flatnonzero(near);world=world[ids]
  frame_yes=np.zeros(len(ids),bool);frame_seen=np.zeros(len(ids),bool);frame_no=np.zeros(len(ids),bool)
  cloud=lidar[:,3:6]@pose[:3,:3].T+pose[:3,3]
  for camera in raw['frames'][frame]['camera_images']:
   cid=camera['camera'];row=catalog[(frame,cid)];cal=source['cameras'][str(cid)];capture=world_from_raw@np.asarray(camera['pose']);masks=[]
   for field in ('file','confidence_file','margin_file'):
    path=bundle/'semantic19'/row[field];verify({'path':str(path),'sha256':row['sha256' if field=='file' else field.replace('_file','_sha256')]});sources.append(asset(path));masks.append(np.asarray(Image.open(path)))
   labels,confidence,margin=masks;h,w=labels.shape;pixels,depth,valid=camera_projection(world,capture,cal,(w,h));lp,ld,lv=camera_projection(cloud,capture,cal,(w,h))
   zbuffer=np.full(h*w,np.inf,np.float32);np.minimum.at(zbuffer,lp[lv,1]*w+lp[lv,0],ld[lv]);zbuffer=minimum_filter(zbuffer.reshape(h,w),size=5,mode='constant',cval=np.inf)
   u,v=pixels.T;valid&=(depth<=zbuffer[v,u]+PARAMS['occlusion_tolerance_m']);valid&=(confidence[v,u]>=PARAMS['semantic_confidence_u8'])&(margin[v,u]>=PARAMS['semantic_margin_u8'])
   valid&=(labels[v,u]<=9) # Dynamic classes and sky never vote free or non-road.
   frame_yes|=valid&(labels[v,u]==0);frame_no|=valid&(labels[v,u]!=0);frame_seen|=valid
  # Conflicting camera views are unknown, not a majority shortcut.
  conflict=frame_yes&frame_no;qualified=frame_seen&~conflict
  seen.ravel()[ids]+=qualified.astype(np.uint16);road.ravel()[ids]+=(frame_yes&qualified).astype(np.uint16)
  receipts.append({'frame':frame,'ground_plane':plane.tolist(),'ground_plane_95pct_residual_m':residual,'road_cells':int((frame_yes&qualified).sum()),'conflicting_cells_unknown':int(conflict.sum())})
  print('MAP',scene,frame,'qualified road cells',int((frame_yes&qualified).sum()),flush=True)
 known=(road>=PARAMS['min_independent_frames'])&(road>=PARAMS['min_road_fraction']*np.maximum(seen,1));poly=polygon_from_mask(known,low,step)
 np.savez_compressed(folder/'support.npz',road_votes=road,qualified_frame_votes=seen,known_road=known,origin=low,step=step)
 result={'schema':'sensor_static_road_domain_v1','scene':scene,'coordinate_system':'first_waymo_ego_forward_left_up_meters','geometry':mapping(poly),'uncertainty_m':PARAMS['uncertainty_m'],'online_static_map_allowed':True,'online_actor_truth_allowed':False,'online_future_ego_trajectory_allowed':False,'raw_HD_map_available':False,'physical_boundary_certified':False,'boundary_kind':'multi-frame calibrated semantic ground support; unknown and holes excluded','white_line_boundary_claim':False,'unknown_is_drivable':False,'hole_filling_applied':False,'parameters':PARAMS,'source_frames':frames,'sources':sources,'support':asset(folder/'support.npz')}
 write(file,result);write(folder/'construction.json',{'frames':receipts,'known_area_m2':poly.area,'no_recorded_route_width_used':True})
 plot(folder,result,poses)
 return result

def plot(folder,result,poses):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,7));poly=shape(result['geometry']);parts=list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
 for part in parts:
  x,y=part.exterior.xy;ax.fill(x,y,color='green',alpha=.3)
  for hole in part.interiors:x,y=hole.xy;ax.fill(x,y,color='white')
 ax.plot(poses[:,0,3],poses[:,1,3],color='blue',label='Recorded path: coverage review only');ax.axis('equal');ax.legend();ax.set_title('Observed road support; blank = unknown/non-road, no HD/white-line claim');fig.savefig(Path(folder)/'map_review.png',dpi=140);plt.close(fig)
