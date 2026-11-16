"""Join source lidar and image semantics. Unobserved points remain unknown."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import argparse,json
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from scene_data import CV_TO_SR,read_lidar,sha,write_json,sr_to_native,ply
UNKNOWN=255

def project(points,extrinsic,intrinsic,wh):
 cv_to_ego=np.eye(4);cv_to_ego[:3,:3]=CV_TO_SR;camera_to_ego=np.asarray(extrinsic)@cv_to_ego;inv=np.linalg.inv(camera_to_ego);pc=np.asarray(points)@inv[:3,:3].T+inv[:3,3]
 k=np.asarray(intrinsic,dtype=float)
 safe_z=np.where(np.abs(pc[:,2])>1e-12,pc[:,2],np.nan);x=pc[:,0]/safe_z;y=pc[:,1]/safe_z;r2=x*x+y*y;radial=1+k[4]*r2+k[5]*r2*r2+k[8]*r2*r2*r2
 xd=x*radial+2*k[6]*x*y+k[7]*(r2+2*x*x);yd=y*radial+k[6]*(r2+2*y*y)+2*k[7]*x*y
 uv=np.c_[k[0]*xd+k[2],k[1]*yd+k[3]]
 valid=np.isfinite(uv).all(axis=1)&(pc[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<wh[0])&(uv[:,1]>=0)&(uv[:,1]<wh[1])
 pix=np.zeros((len(pc),2),dtype=np.int32);pix[valid]=np.floor(uv[valid]).astype(np.int32)
 return pix,pc[:,2],valid

def labels_for_points(points,cameras,masks,wh,tolerance):
 votes=np.zeros((len(points),19),dtype=np.uint8)
 for cam,mask in masks.items():
  c=cameras[str(cam)];pixels,z,valid=project(points,c['camera_waymo_to_ego'],c['intrinsics_9'],wh[str(cam)]);idx=np.flatnonzero(valid)
  if not len(idx):continue
  h,w=mask.shape;depth=np.full((h,w),np.inf,dtype=np.float32);uv=pixels[idx];np.minimum.at(depth,(uv[:,1],uv[:,0]),z[idx]);near=minimum_filter(depth,size=3)
  idx=idx[z[idx]<=near[uv[:,1],uv[:,0]]+tolerance];uv=pixels[idx];labels=mask[uv[:,1],uv[:,0]]
  qualified=labels<19;np.add.at(votes,(idx[qualified],labels[qualified]),1)
 maximum=votes.max(axis=1);label=votes.argmax(axis=1).astype(np.uint8);tie=(votes==maximum[:,None]).sum(axis=1)>1;label[(maximum==0)|tie]=UNKNOWN
 return label,maximum

def physical_lidar_labels(labels,geometric_ground):
 labels=np.asarray(labels).copy();sky=labels==10;unsupported_road=(labels==0)&~np.asarray(geometric_ground,dtype=bool)
 labels[sky|unsupported_road]=UNKNOWN
 return labels,{'projected_sky_lidar_points':int(sky.sum()),'road_without_geometric_ground_points':int(unsupported_road.sum())}

def background_frame_labels(background,world,labels,max_distance):
 background=np.asarray(background);world=np.asarray(world);labels=np.asarray(labels)
 if not len(world):return np.empty(0,dtype=int),np.empty(0,dtype=np.uint8)
 low=world.min(axis=0)-max_distance;high=world.max(axis=0)+max_distance;candidate=np.flatnonzero(np.all((background>=low)&(background<=high),axis=1))
 if not len(candidate):return candidate,np.empty(0,dtype=np.uint8)
 distance,index=cKDTree(world).query(background[candidate],k=2,distance_upper_bound=max_distance)
 first=np.isfinite(distance[:,0]);safe=np.minimum(index,len(world)-1);chosen=labels[safe[:,0]];second=np.isfinite(distance[:,1]);conflict=second&(labels[safe[:,1]]!=chosen)&(distance[:,1]<=1.5*distance[:,0]+.02)
 valid=first&~conflict&(chosen!=UNKNOWN)
 return candidate[valid],chosen[valid]

def transfer(bundle,max_distance=.25,occlusion_tolerance=.5):
 bundle=Path(bundle);meta=_hugsim_expand(json.loads((bundle/'source_manifest.json').read_text()));sem=bundle/'semantic19';sm=_hugsim_expand(json.loads((sem/'semantic_manifest.json').read_text()))
 if sm['source_checkpoint_sha256']!=meta['source_checkpoint_sha256'] or len(sm['maps'])!=meta['frames']*5:raise ValueError('Wrong or incomplete semantic maps')
 for row in sm['maps']:
  if sha(sem/row['file'])!=row['sha256']:raise ValueError('Changed semantic map')
  for key in ['confidence','margin']:
   if sha(sem/row[key+'_file'])!=row[key+'_sha256']:raise ValueError('Changed semantic confidence')
 out=bundle/'point_semantics';out.mkdir(exist_ok=False);poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];bg=np.load(bundle/'background_sr_world.npy',mmap_mode='r');votes=np.zeros((len(bg),19),dtype=np.uint16);records=[]
 raw=_hugsim_expand(json.loads((bundle/'raw_metadata/raw_metadata.json').read_text()));raw_frames={x['frame']:x for x in raw['frames']}
 for row in meta['frames_data']:
  i=row['frame'];source=Path(row['lidar']['path'])
  if sha(source)!=row['lidar']['sha256']:raise ValueError('Changed lidar')
  lidar=read_lidar(source);masks={}
  for cam in range(5):
   name='%03d_%d'%(i,cam);mask=np.asarray(Image.open(sem/(name+'.png'))).copy();confidence=np.asarray(Image.open(sem/(name+'_confidence.png')));margin=np.asarray(Image.open(sem/(name+'_margin.png')));mask[(confidence<128)|(margin<26)]=UNKNOWN;masks[cam]=mask
  cameras={k:dict(v) for k,v in meta['cameras'].items()};frame_pose=np.loadtxt(Path(meta['raw_data'])/'ego_pose'/('%03d.txt'%i))
  for camera in raw_frames[i]['camera_images']:
   k=str(camera['camera']);cameras[k]['camera_waymo_to_ego']=(np.linalg.inv(frame_pose)@np.asarray(camera['pose'])@np.asarray(cameras[k]['camera_waymo_to_ego'])).tolist()
  labels,confidence=labels_for_points(lidar[:,3:6],cameras,masks,row['image_size_wh'],occlusion_tolerance)
  labels,contradictions=physical_lidar_labels(labels,lidar[:,10])
  np.savez_compressed(out/('%03d.npz'%i),class_id=labels,agreeing_camera_votes=confidence)
  # Moving actor hits never label the static reconstruction. Road is kept separately.
  valid=(labels!=UNKNOWN)&(lidar[:,9]==0);local=np.asarray(lidar[valid,3:6]);world=local@poses[i,:3,:3].T+poses[i,:3,3];index,assigned=background_frame_labels(bg,world,labels[valid],max_distance)
  if len(index):np.add.at(votes,(index,assigned),1)
  records.append({'frame':i,'points':len(labels),'labeled_points':int(np.sum(labels!=UNKNOWN)),'unknown_points':int(np.sum(labels==UNKNOWN)),'background_gaussians_with_frame_support':len(index),'contradictions_kept_unknown':contradictions})
  if i%20==0:print('Point semantics frame',i,flush=True)
 maxima=votes.max(axis=1);class_id=votes.argmax(axis=1).astype(np.uint8);ambiguous=(votes==maxima[:,None]).sum(axis=1)!=1;class_id[(maxima<2)|ambiguous]=UNKNOWN
 np.savez_compressed(out/'background_semantics.npz',class_id=class_id,frame_votes=votes)
 selected=(class_id>1)&(class_id!=10)&(class_id!=UNKNOWN);ply(out/'classified_scene.ply',sr_to_native(bg[selected]));ply(out/'unobserved_background.ply',sr_to_native(bg[class_id==UNKNOWN]))
 result={'schema':'source_point_semantics_v1','frames':records,'background_points':len(bg),'class_histogram':{str(k):int(np.sum(class_id==k)) for k in [*range(19),UNKNOWN]},'unknown_class':UNKNOWN,'transfer_max_distance_m':max_distance,'occlusion_depth_tolerance_m':occlusion_tolerance,'minimum_independent_frame_votes':2,'semantic_top1_minimum_proxy':128/255,'semantic_top1_top2_margin_minimum_proxy':26/255,'proxy_thresholds_do_not_certify_physical_truth':True,'lidar_columns_preserved_in_source':True,'unknown_is_free':False,'semantic_pointcloud_source':'recorded original ego path, not synthetic closed-loop sensor','native_evaluation_admitted':False,'requires_review':['unobserved and ambiguous background coverage','point-density-dependent native collision threshold','trained box versus physical geometry','source 10Hz assumption'],'source_sha256':sha(Path(__file__)),'background_transfer_direction':'each Gaussian queries nearby qualified source lidar; one vote per Gaussian per frame','cross_class_nearest_conflicts_kept_unknown':True,'per_camera_capture_pose_used':True,'rolling_shutter_per_row_compensation':False,'ground_truth_used_by_optimizer':False}
 write_json(out/'point_semantics_manifest.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();transfer(a.bundle)
if __name__=='__main__':main()
