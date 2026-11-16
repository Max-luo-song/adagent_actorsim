"""Trace retained negative votes to original semantic pixels; no relabeling."""
import argparse,collections
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from scene_contracts import *
from boundaries import camera_projection

def run(scene):
 diagnostic=DIAGNOSTICS/'start_support_v1'/str(scene)/'report.json';audit=read(diagnostic);cells=[c for c in audit['cells'] if c['nonroad_votes']>0]
 if not cells:return {'scene':scene,'retained_negative_cells':0,'retained_classes':[]}
 bundle=ASSETS/str(scene);source=read(bundle/'source_manifest.json');raw=read(bundle/'raw_metadata/raw_metadata.json');semantic=read(bundle/'semantic19/semantic_manifest.json');catalog={(r['frame'],r['camera']):r for r in semantic['maps']};data=read(audit['map']['path']);params=data['parameters'];poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];origin=np.linalg.inv(np.loadtxt(Path(source['raw_data'])/'ego_pose/000.txt'));planes={r['frame']:np.asarray(r['ground_plane']) for r in read(MAPS/str(scene)/'construction.json')['frames']};xy=np.array([[c['x'],c['y']] for c in cells]);N=len(cells);counts=np.zeros(N,int);events=[];ignored=collections.Counter()
 for frame in data['source_frames']:
  pose=poses[frame];plane=planes[frame];normal=pose[:3,:3]@np.r_[-plane[:2],1.];offset=plane[2]+normal@pose[:3,3];z=(offset-normal[0]*xy[:,0]-normal[1]*xy[:,1])/normal[2];world=np.c_[xy,z];local=(world-pose[:3,3])@pose[:3,:3];near=np.linalg.norm(local[:,:2],axis=1)<=params['max_local_range_m'];views=[]
  for camera in raw['frames'][frame]['camera_images']:
   cid=camera['camera'];cal=source['cameras'][str(cid)];capture=origin@np.asarray(camera['pose']);row=catalog[(frame,cid)];wh=source['frames_data'][frame]['image_size_wh'][str(cid)];uv,depth,valid=camera_projection(world,capture,cal,wh);valid&=near
   if valid.any():views.append((cid,cal,capture,row,uv,depth,valid))
  if not views:continue
  a=source['frames_data'][frame]['lidar'];verify(a);lidar=np.memmap(a['path'],dtype='<f4',mode='r').reshape(-1,14);cloud=lidar[:,3:6]@pose[:3,:3].T+pose[:3,3];yes=np.zeros(N,bool);no=np.zeros(N,bool);frame_events=[]
  for cid,cal,capture,row,uv,depth,visible in views:
   masks=[]
   for key in ('file','confidence_file','margin_file'):
    file=bundle/'semantic19'/row[key];verify({'path':str(file),'sha256':row['sha256' if key=='file' else key.replace('_file','_sha256')]});masks.append(np.asarray(Image.open(file)))
   labels,confidence,margin=masks;h,w=labels.shape;lp,ld,lv=camera_projection(cloud,capture,cal,(w,h));buffer=np.full(h*w,np.inf,np.float32);np.minimum.at(buffer,lp[lv,1]*w+lp[lv,0],ld[lv]);buffer=minimum_filter(buffer.reshape(h,w),size=5,mode='constant',cval=np.inf);u,v=uv.T;valid=visible&(depth<=buffer[v,u]+params['occlusion_tolerance_m'])&(confidence[v,u]>=params['semantic_confidence_u8'])&(margin[v,u]>=params['semantic_margin_u8']);classes=labels[v,u]
   for cls,n in zip(*np.unique(classes[valid&(classes>=10)],return_counts=True)):ignored[int(cls)]+=int(n)
   static=valid&(classes<=9);yes|=static&(classes==0);no|=static&(classes!=0)
   for index in np.flatnonzero(static&(classes!=0)):
    image=source['frames_data'][frame]['camera_files'][str(cid)+':image'];frame_events.append({'cell_index':int(index),'x':cells[index]['x'],'y':cells[index]['y'],'body_area_m2':cells[index]['body_area_m2'],'ring_area_m2':cells[index]['ring_area_m2'],'frame':frame,'camera':cid,'u':int(u[index]),'v':int(v[index]),'class_id':int(classes[index]),'class_name':semantic['classes'][int(classes[index])],'confidence_u8':int(confidence[v[index],u[index]]),'margin_u8':int(margin[v[index],u[index]]),'ground_query_world_z':float(z[index]),'pixel_distance_to_image_border':int(min(u[index],v[index],w-1-u[index],h-1-v[index])),'source_image':image})
  retained=no&~yes;counts+=retained;events.extend(e for e in frame_events if retained[e['cell_index']])
 if not np.array_equal(counts,[c['nonroad_votes'] for c in cells]):raise ValueError('Negative pixel trace does not reproduce frozen retained votes')
 histogram=collections.Counter(e['class_name'] for e in events);result={'schema':'negative_vote_pixel_provenance_v1','scene':scene,'diagnostic':asset(diagnostic),'retained_negative_cells':len(cells),'retained_classes':sorted(histogram),'negative_pixel_event_histogram':dict(histogram),'histogram_is_pixel_provenance_not_independent_frame_votes':True,'ignored_sky_dynamic_or_unknown_pixel_events':dict(ignored),'explicit_ego_pixel_mask_present':False,'dynamic_class_ids_11_to_18_counted_as_nonroad':False,'events':events,'map_or_vote_policy_modified':False,'GPU_executed':False}
 folder=DIAGNOSTICS/'negative_pixels_v1'/str(scene);write(folder/'report.json',result);plot(events,folder);return result

def plot(events,folder):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 chosen=[];groups=set()
 for event in sorted(events,key=lambda e:(e['body_area_m2']<=0,e['frame'],e['camera'])):
  key=(event['frame'],event['camera'])
  if key not in groups:chosen.append(event);groups.add(key)
  if len(chosen)==6:break
 fig,axes=plt.subplots(2,3,figsize=(12,8));
 for ax,event in zip(axes.flat,chosen):
  verify(event['source_image']);rgb=np.asarray(Image.open(event['source_image']['path']));u,v=event['u'],event['v'];pad=90;x0=max(0,u-pad);x1=min(rgb.shape[1],u+pad+1);y0=max(0,v-pad);y1=min(rgb.shape[0],v+pad+1);ax.imshow(rgb[y0:y1,x0:x1]);ax.plot(u-x0,v-y0,'r+',markersize=12);ax.set_title('f%d cam%d: %s\ncell=(%.1f,%.1f)'%(event['frame'],event['camera'],event['class_name'],event['x'],event['y']));ax.axis('off')
 for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
 fig.suptitle('Retained negative-vote pixels; source images, not physical ground truth');fig.tight_layout();fig.savefig(folder/'source_pixel_review.png',dpi=140);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);a=p.parse_args();r=run(a.scene);print({k:v for k,v in r.items() if k!='events'})
