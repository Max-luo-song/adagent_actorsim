import ast,types,sys,collections
import numpy as np
from PIL import Image,ImageDraw
from common import *

def parser():
 source=ROOT/'integrations/patch-generalization-assets-v1/raw_metadata.py'
 tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) and (not isinstance(n,ast.FunctionDef) or n.name in ('varint','fields','transform','records','label')) and not (isinstance(n,ast.ImportFrom) and n.module=='scene_data')]
 ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);return ns

def raw_meta(scene,report,raw):
 cached=OLD_ASSETS/str(scene)/'raw_metadata/raw_metadata.json'
 if cached.exists():
  x=read(cached)
  if x['scene_index']!=scene or not x['raw_processed_pose_identity_all_frames']:raise ValueError('Source mismatch')
  return x,asset(cached)
 pr=parser();listing=OMNI/'data/waymo_train_list.txt';name=listing.read_text().splitlines()[scene]
 path=OMNI.parent/'data/waymo/raw'/(name+'.tfrecord');rows=[]
 for i,(offset,data) in enumerate(pr['records'](path)):
  f=pr['fields'](data);pose=pr['transform'](f[3][0]);expected=np.loadtxt(raw/'ego_pose'/('%03d.txt'%i))
  if not np.allclose(pose,expected,atol=1e-8,rtol=0):raise ValueError('Raw source pose identity mismatch')
  cams=[]
  for encoded in f.get(4,[]):
   c=pr['fields'](encoded);cid=int(c[1][0])-1
   if __import__('hashlib').sha256(c[2][0]).hexdigest()!=sha(raw/'images'/('%03d_%d.jpg'%(i,cid))):raise ValueError('Raw image identity mismatch')
   cams.append({'camera':cid,'pose':pr['transform'](c[3][0]).tolist()})
  rows.append({'frame':i,'timestamp_micros':int(f[2][0]),'camera_images':cams,'object_labels':[pr['label'](v) for v in f.get(6,[])]})
  if i%30==0:print('RAW',scene,i,flush=True)
 if len(rows)!=report['frames']:raise ValueError('Raw frame count mismatch')
 x={'scene_index':scene,'raw_record':str(path),'raw_record_sha256':sha(path),'raw_processed_pose_identity_all_frames':True,'raw_processed_image_identity_all_images':True,'raw_processed_calibration_identity':False,'frames':rows}
 # Explicitly compare raw context calibration in first record.
 _,data=next(pr['records'](path));context=pr['fields'](pr['fields'](data)[1][0])
 for encoded in context.get(2,[]):
  c=pr['fields'](encoded);cid=int(c[1][0])-1;intr=c[2]
  if len(intr)==1 and isinstance(intr[0],bytes):intr=np.frombuffer(intr[0],dtype='<f8')
  if not np.allclose(intr,np.loadtxt(raw/'intrinsics'/('%d.txt'%cid)),atol=1e-8,rtol=0):raise ValueError('Raw intrinsics mismatch')
  if not np.allclose(pr['transform'](c[3][0]),np.loadtxt(raw/'extrinsics'/('%d.txt'%cid)),atol=1e-8,rtol=0):raise ValueError('Raw extrinsics mismatch')
 x['raw_processed_calibration_identity']=True
 return x,asset(path)

def smooth(x,n=5):return np.convolve(np.pad(x,(n//2,n//2),mode='edge'),np.ones(n)/n,mode='valid')
def episodes(mask,minimum=5):
 e=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
 return [(int(a),int(b-1)) for a,b in e.reshape(-1,2) if b-a>=minimum]

def core(meta,poses):
 rule=read(OUT/'preregistration.json')['core_rule'];t=np.array([r['timestamp_micros'] for r in meta['frames']],float)/1e6;t-=t[0]
 if not np.all(np.diff(t)>0):raise ValueError('Non-monotonic raw clock')
 xy=poses[:,:2,3];v=np.linalg.norm(np.gradient(xy,t,axis=0),axis=1);a=np.gradient(smooth(v),t);yaw=np.unwrap(np.arctan2(poses[:,1,0],poses[:,0,0]));rate=np.gradient(smooth(yaw),t)
 event=[]
 for kind,mask in [('ego_turn',(abs(rate)>=rule['yaw_rate_threshold_rad_s'])&(v>=rule['ego_speed_for_turn_mps'])),('ego_brake',a<=rule['ego_deceleration_threshold_mps2'])]:
  event.extend({'kind':kind,'start':b,'end':e} for b,e in episodes(mask,rule['minimum_sustained_frames']))
 # Original labels are in each frame's ego coordinates; closing gap derivative is ego-relative.
 tracks=collections.defaultdict(dict)
 for i,row in enumerate(meta['frames']):
  for obj in row['object_labels']:
   if obj['waymo_type'] in (1,2,4):tracks[obj['id']][i]=obj
 min_ttc=None
 for key,track in tracks.items():
  ids=np.array(sorted(track));local=np.array([track[int(i)]['center_vehicle'][:2] for i in ids]);long=local[:,0]
  closing=-np.gradient(smooth(long),t[ids]) if len(ids)>4 else np.zeros(len(ids))
  gap=np.array([long[j]-track[int(i)]['size_lwh'][0]/2-1.5 for j,i in enumerate(ids)])
  ttc=np.divide(np.maximum(gap,0),closing,out=np.full(len(ids),np.inf),where=closing>.1)
  valid=(gap>0)&(gap<=40)&(abs(local[:,1])<=2.5)&(ttc<=4)
  dense=np.zeros(len(t),bool);dense[ids]=valid
  for b,e in episodes(dense,rule['minimum_sustained_frames']):
   value=float(np.min(ttc[(ids>=b)&(ids<=e)]));min_ttc=value if min_ttc is None else min(min_ttc,value)
   event.append({'kind':'actor_closing_corridor','actor_id':key,'start':b,'end':e,'min_raw_kinematic_ttc_s':value})
 interval=None
 if event:
  st=max(0.,t[min(e['start'] for e in event)]-rule['padding_seconds']);en=min(t[-1],t[max(e['end'] for e in event)]+rule['padding_seconds'])
  interval=[int(np.searchsorted(t,st,side='right')-1),int(np.searchsorted(t,en,side='left'))]
 return {'core_interval_frames':interval,'events':event,'topology':sorted(set(e['kind'] for e in event)),'severity':'high' if (min_ttc is not None and min_ttc<=2) or (len(event)>0 and float(a.min())<=-3) else 'moderate','min_raw_kinematic_ttc_s':min_ttc,'minimum_smoothed_ego_acceleration_mps2':float(a.min()),'timestamp_seconds_from_start':t.tolist(),'raw_clock_authority':True,'core_rule_sha256':sha(OUT/'preregistration.json')}

def prepare(scene):
 report=read(REPORTS[scene]);old=OLD_ASSETS/str(scene)/'source_manifest.json';raw=Path(read(old)['raw_data'] if old.exists() else report.get('data_dir',report.get('original_data')));bundle=ASSETS/str(scene);bundle.mkdir(parents=True,exist_ok=True)
 fs=sorted((raw/'ego_pose').glob('*.txt'));poses=np.stack([np.loadtxt(f) for f in fs]);relative=np.linalg.inv(poses[0])@poses
 mp=bundle/'raw_metadata/raw_metadata.json'
 if mp.exists():meta=read(mp)
 else:meta,receipt=raw_meta(scene,report,raw);meta['upstream_metadata_receipt']=receipt;write(mp,meta)
 cp=bundle/'core_interval.json'
 if not cp.exists():freeze(cp,core(meta,poses))
 core_data=read(cp);t=np.array(core_data['timestamp_seconds_from_start']);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(poses[:,:2,3],axis=0),axis=1))]
 np.savez_compressed(bundle/'poses_and_clock.npz',relative_ego_to_sr_world=relative,timestamp_raw_s=t,cumulative_travel_m=arc)
 candidates=np.flatnonzero((arc>=6)&(t[-1]-t>=10))
 interval=core_data['core_interval_frames'];retained=[int(i) for i in candidates if interval is not None and i<=interval[0]]
 write(bundle/'necessary_admission.json',{'scene':scene,'frames':len(poses),'total_travel_m':float(arc[-1]),'first_6m_frame':int(np.flatnonzero(arc>=6)[0]) if np.any(arc>=6) else None,'first_6m_remaining_seconds':float(t[-1]-t[np.flatnonzero(arc>=6)[0]]) if np.any(arc>=6) else None,'window_qualified_frames':candidates.tolist(),'core_preserving_candidates':retained,'core_interval':interval,'map_gate':'not_yet_run','necessary_conditions_pass':bool(retained)})
 # Minimal original-sensor manifest, no reconstruction or route-derived road width.
 cameras={};frames=[];basis=[[0,0,1],[-1,0,0],[0,-1,0]]
 for cam in range(5):cameras[str(cam)]={'intrinsics_9':np.loadtxt(raw/'intrinsics'/('%d.txt'%cam)).tolist(),'camera_waymo_to_ego':np.loadtxt(raw/'extrinsics'/('%d.txt'%cam)).tolist(),'opencv_to_waymo':basis}
 sampled=list(range(0,len(poses),5));sampled+=[] if sampled[-1]==len(poses)-1 else [len(poses)-1]
 for i in range(len(poses)):
  files={};wh={}
  for cam in range(5):
   path=raw/'images'/('%03d_%d.jpg'%(i,cam));files[str(cam)+':image']=asset(path)
   with Image.open(path) as im:wh[str(cam)]=list(im.size)
  lp=raw/'lidar'/('%03d.bin'%i)
  frames.append({'frame':i,'camera_files':files,'image_size_wh':wh,'lidar':asset(lp)})
 write(bundle/'source_manifest.json',{'schema':'static_original_sensor_only_v1','scene':scene,'frames':len(poses),'raw_data':str(raw),'frames_data':frames,'cameras':cameras,'source_report':asset(REPORTS[scene]),'sampled_frames':sampled,'no_reconstruction_geometry_used':True})
 print('PREPARED',scene,'core',interval,'necessary starts',len(retained),flush=True)

def overview():
 canvas=Image.new('RGB',(5*480,7*205),'#eeeeee');draw=ImageDraw.Draw(canvas)
 for row,scene in enumerate(SCENES):
  raw=Path(read(ASSETS/str(scene)/'source_manifest.json')['raw_data'])
  for cam in range(5):
   with Image.open(raw/'images'/('000_%d.jpg'%cam)) as im:
    w,h=im.size;crop=im.crop((0,int(h*.65),w,h));crop.thumbnail((480,180));canvas.paste(crop,(cam*480,row*205+25))
   draw.text((cam*480+5,row*205+5),'scene%d camera%d bottom35%%'%(scene,cam),fill='black')
 canvas.save(OUT/'ego_mask_source_review.jpg')

if __name__=='__main__':
 for s in SCENES:prepare(s)
 freeze(OUT/'core_intervals_frozen.json',{'preregistration':asset(OUT/'preregistration.json'),'scenes':{str(s):asset(ASSETS/str(s)/'core_interval.json') for s in SCENES},'new_map_outputs_read_before_freeze':False})
 overview()
