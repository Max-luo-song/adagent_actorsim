"""Read selected Waymo protobuf fields without a TensorFlow dependency.
Field contract: official waymo-open-dataset dataset.proto and label.proto.
Unknown fields are preserved/skipped by wire type, never assigned semantics.
"""

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

import argparse,hashlib,json,struct,zlib
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_data import RAW,OMNI,sha,write_json
SCHEMA_URL='https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/dataset.proto'
def varint(data,index):
 value=0
 for shift in range(0,70,7):
  if index>=len(data):raise ValueError('Truncated protobuf varint')
  b=data[index];index+=1;value|=(b&127)<<shift
  if b<128:return value,index
 raise ValueError('Oversized protobuf varint')
def fields(data):
 result={};i=0
 while i<len(data):
  key,i=varint(data,i);field,wire=key>>3,key&7
  if field==0:raise ValueError('Invalid protobuf field zero')
  if wire==0:value,i=varint(data,i)
  elif wire in (1,5):
   n=8 if wire==1 else 4
   if i+n>len(data):raise ValueError('Truncated protobuf scalar')
   value=struct.unpack('<d' if wire==1 else '<f',data[i:i+n])[0];i+=n
  elif wire==2:
   n,i=varint(data,i)
   if i+n>len(data):raise ValueError('Truncated protobuf bytes')
   value=data[i:i+n];i+=n
  else:raise ValueError('Unsupported protobuf wire type %d'%wire)
  result.setdefault(field,[]).append(value)
 return result

def transform(data):
 values=fields(data).get(1,[])
 if len(values)==1 and isinstance(values[0],bytes):values=np.frombuffer(values[0],dtype='<f8')
 result=np.asarray(values,dtype=float)
 if result.size!=16 or not np.isfinite(result).all():raise ValueError('Invalid Waymo transform')
 return result.reshape(4,4)

def records(path):
 with Path(path).open('rb') as f:
  while True:
   offset=f.tell();header=f.read(12)
   if not header:return
   if len(header)!=12:raise ValueError('Truncated TFRecord header')
   n=struct.unpack('<Q',header[:8])[0]
   if n>128*1024*1024:raise ValueError('Unexpected TFRecord length')
   data=f.read(n);crc=f.read(4)
   if len(data)!=n or len(crc)!=4:raise ValueError('Truncated TFRecord data')
   yield offset,data

def label(data):
 item=fields(data);box=fields(item[1][0]);dims=[box[k][0] for k in [5,4,6]]
 if not np.isfinite(dims).all() or min(dims)<=0:raise ValueError('Bad source object dimensions')
 return {'id':item[4][0].decode(),'waymo_type':int(item[3][0]),'center_vehicle':[box[k][0] for k in [1,2,3]],'size_lwh':dims,'heading_vehicle':box[7][0],'velocity_acceleration':{str(k):v[0] for k,v in fields(item.get(2,[b''])[0]).items()}}

def export(bundle):
 bundle=Path(bundle);meta=_hugsim_expand(json.loads((bundle/'source_manifest.json').read_text()));listing=OMNI/'data/waymo_train_list.txt';name=listing.read_text().splitlines()[551];raw=RAW.parents[2]/'raw'/(name+'.tfrecord')
 # RAW = processed/training/551, parents[2] is waymo.
 if not raw.exists():raise FileNotFoundError(raw)
 out=bundle/'raw_metadata';out.mkdir(exist_ok=False);rows=[];timestamp=[];first_pose=np.loadtxt(RAW/'ego_pose/000.txt');camera_seg=0;lidar_seg=0;map_features=0;label_count=0
 for i,(offset,data) in enumerate(records(raw)):
  if i>=meta['frames']:raise ValueError('Raw/processed frame count mismatch')
  frame=fields(data);pose=transform(frame[3][0]);source_pose=np.loadtxt(RAW/'ego_pose'/('%03d.txt'%i))
  if not np.allclose(pose,source_pose,atol=1e-8,rtol=0):raise ValueError('Raw pose mismatch at frame '+str(i))
  stamp=int(frame[2][0]);timestamp.append(stamp);images=[];seen=set();context=fields(frame[1][0]);lasers=[]
  for encoded in frame.get(4,[]):
   camera=fields(encoded);cam=int(camera[1][0])-1;seen.add(cam);im=camera[2][0];src=meta['frames_data'][i]['camera_files'][str(cam)+':image']
   if hashlib.sha256(im).hexdigest()!=src['sha256']:raise ValueError('Raw image mismatch')
   segmentation=camera.get(10,[])
   if segmentation:
    camera_seg+=1;(out/('%03d_cam%d_segmentation.pb'%(i,cam))).write_bytes(segmentation[0])
   images.append({'camera':cam,'image_bitwise_equal':True,'pose':transform(camera[3][0]).tolist(),'pose_timestamp_unix_s':camera.get(5,[None])[0],'shutter_s':camera.get(6,[None])[0],'trigger_unix_s':camera.get(7,[None])[0],'readout_done_unix_s':camera.get(8,[None])[0],'velocity':{str(k):v[0] for k,v in fields(camera.get(4,[b''])[0]).items()},'has_original_semantics':bool(segmentation)})
  if seen!=set(range(5)):raise ValueError('Raw camera coverage mismatch')
  for encoded in frame.get(5,[]):
   laser=fields(encoded);lid=int(laser[1][0]);returns=[]
   for field in [2,3]:
    if field not in laser:continue
    ri=fields(laser[field][0]);has_seg=6 in ri
    if has_seg:
     lidar_seg+=1;(out/('%03d_laser%d_return%d_segmentation.zlib'%(i,lid,field-1))).write_bytes(ri[6][0])
    returns.append({'return':field-1,'has_segmentation':has_seg,'camera_projection_present':3 in ri,'range_pose_present':4 in ri})
   lasers.append({'name':lid,'returns':returns})
  labels=[label(v) for v in frame.get(6,[])];label_count+=len(labels);map_features+=len(frame.get(10,[]))
  # Keep original calibration messages, including rolling-shutter and beam information.
  if i==0:
   (out/'context.pb').write_bytes(frame[1][0]);calibration={'cameras':[],'lasers':[]}
   for encoded in context.get(2,[]):
    c=fields(encoded);intrinsic=c.get(2,[])
    if len(intrinsic)==1 and isinstance(intrinsic[0],bytes):intrinsic=np.frombuffer(intrinsic[0],dtype='<f8').tolist()
    cam=int(c[1][0])-1
    if not np.allclose(intrinsic,meta['cameras'][str(cam)]['intrinsics_9'],atol=1e-8):raise ValueError('Raw intrinsic mismatch')
    if not np.allclose(transform(c[3][0]),meta['cameras'][str(cam)]['camera_waymo_to_ego'],atol=1e-8):raise ValueError('Raw extrinsic mismatch')
    calibration['cameras'].append({'camera':cam,'intrinsics_9':intrinsic,'extrinsic':transform(c[3][0]).tolist(),'width':c.get(4,[None])[0],'height':c.get(5,[None])[0],'rolling_shutter_direction':c.get(6,[None])[0]})
   for encoded in context.get(3,[]):
    c=fields(encoded);incl=c.get(2,[])
    if len(incl)==1 and isinstance(incl[0],bytes):incl=np.frombuffer(incl[0],dtype='<f8').tolist()
    calibration['lasers'].append({'name':c[1][0],'beam_inclinations':incl,'inclination_min':c.get(3,[None])[0],'inclination_max':c.get(4,[None])[0],'extrinsic':transform(c[5][0]).tolist()})
   write_json(out/'calibration.json',calibration)
  rows.append({'frame':i,'raw_offset':offset,'timestamp_micros':stamp,'camera_images':images,'lasers':lasers,'object_labels':labels,'map_feature_count':len(frame.get(10,[]))})
  if i%25==0:print('Raw identity verified frame',i,flush=True)
 if len(rows)!=meta['frames'] or np.any(np.diff(timestamp)<=0):raise ValueError('Invalid source clock coverage')
 stamp=np.asarray(timestamp,dtype=np.int64);elapsed=(stamp-stamp[0])/1e6;assumed=np.arange(len(stamp))*.1
 np.savez(out/'hardware_clock.npz',timestamp_micros=stamp,elapsed_s=elapsed,simulation_clock_s=assumed,recorded_minus_uniform_s=elapsed-assumed)
 result={'schema':'waymo551_raw_metadata_v1','raw_record':str(raw),'raw_record_sha256':sha(raw),'training_list_sha256':sha(listing),'scene_index':551,'protobuf_schema':SCHEMA_URL,'raw_processed_pose_identity_all_frames':True,'raw_processed_image_identity_all_995_images':True,'raw_processed_calibration_identity':True,'tfrecord_crc_checked':False,'record_bounds_and_source_identity_checked':True,'frames':rows,'original_camera_semantic_images':camera_seg,'original_lidar_semantic_returns':lidar_seg,'original_map_features':map_features,'original_3d_labels':label_count,'clock':{'authority':'original timestamp_micros plus per-camera pose/exposure times','frame_start_is_not_pose_timestamp':True,'min_frame_dt_s':float(np.diff(stamp).min()/1e6),'max_frame_dt_s':float(np.diff(stamp).max()/1e6),'max_difference_from_uniform10Hz_s':float(np.max(np.abs(elapsed-assumed))),'simulation_clock_policy':'unchanged source adapter frame index/10; hardware times exported explicitly, not silently substituted'},'semantic_taxonomies_must_not_be_conflated':True,'ground_truth_used_by_optimizer':False,'native_evaluation_admitted':False}
 write_json(out/'raw_metadata.json',result);print('Original metadata export:',out,flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();export(a.bundle)
if __name__=='__main__':main()
