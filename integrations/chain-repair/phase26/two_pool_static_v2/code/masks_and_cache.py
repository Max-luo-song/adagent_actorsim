import shutil
import numpy as np
from PIL import Image
from common import *
from ego_mask import rasterize,REVISION

def masks():
 for scene in SCENES:
  b=ASSETS/str(scene);src=read(b/'source_manifest.json');cams={};reviews=[]
  for frame in (0,src['frames']//2,src['frames']-1):
   for cam in range(5):reviews.append(src['frames_data'][frame]['camera_files'][str(cam)+':image'])
  for cam in range(5):
   wh=src['frames_data'][0]['image_size_wh'][str(cam)]
   if any(r['image_size_wh'][str(cam)]!=wh for r in src['frames_data']):raise ValueError('Variable source camera crop')
   path=b/'ego_masks'/('%d.png'%cam);path.parent.mkdir(parents=True,exist_ok=True)
   # Explicit reviewed-empty annotation. This is never a fallback for a missing mask.
   ar=rasterize([],wh,empty_reviewed=True);Image.fromarray(ar).save(path)
   cams[str(cam)]={'authority':'explicit_raw_image_annotation','mask':asset(path),'explicit_reviewed_empty':True,'polygons':[],'review_frames':[0,src['frames']//2,src['frames']-1],'visible_ego_pixels_annotated':int(np.count_nonzero(ar)),'limitation':'No identifiable ego body in reviewed views. Fixed rig/crop extrapolation; not per-frame ground-truth labeling.'}
  freeze(b/'ego_masks/manifest.json',{'revision':REVISION,'scene':scene,'source_manifest':asset(b/'source_manifest.json'),'cameras':cams,'review_contact_sheet':asset(OUT/('ego_review_scene%d.jpg'%scene)),'raw_review_images':reviews,'reviewer':'assistant visual inspection of original-image contact sheets','display_transform':'autocontrast only for review; original image inputs to semantics remain unmodified','low_light_review_limitation':scene==233,'semantic_car_class_not_used_as_ego_mask':True,'unknown_or_nonroad_not_filled':True})
  print('EGO explicit source-pixel masks',scene,flush=True)

def reuse552():
 scene=552;b=ASSETS/str(scene);src=read(b/'source_manifest.json');old=OLD_ASSETS/str(scene)/'semantic19';m=read(old/'semantic_manifest.json');rows=[];dest=b/'semantic19';dest.mkdir(exist_ok=True)
 for original in m['maps']:
  if original['frame'] not in src['sampled_frames']:continue
  image=src['frames_data'][original['frame']]['camera_files'][str(original['camera'])+':image']
  if original['image_sha256']!=image['sha256']:raise ValueError('Cannot reuse semantic prediction for another image')
  verify(image);row={k:original[k] for k in ('frame','camera','image_sha256','inference_contract_sha256','file','sha256','confidence_file','confidence_sha256','margin_file','margin_sha256')}
  for key in ('file','confidence_file','margin_file'):
   f=old/row[key];verify({'path':str(f),'sha256':row['sha256' if key=='file' else key.replace('_file','_sha256')]});shutil.copy2(f,dest/row[key])
  rows.append(row)
 if len(rows)!=len(src['sampled_frames'])*5:raise ValueError('Incomplete reused map-sample coverage')
 write(dest/'semantic_manifest.json',{'schema':'map_sampled_original_semantics_v1','scene':scene,'maps':rows,'sampled_frames':src['sampled_frames'],'coverage':'all5 cameras for every preregistered map sample; full original-time semantics retained upstream','upstream_manifest':asset(old/'semantic_manifest.json'),'source_manifest':asset(b/'source_manifest.json'),'inference_audit':m['inference_audit'],'inference_contract_sha256':m['inference_contract_sha256'],'reused_same_raw_image_frozen_predictions':True,'new_inference_claimed':False,'selfmask_new_release_applied_in_map_not_by_mutating_raw_predictions':True})
 print('552 cached semantic image identities verified',len(rows),flush=True)

if __name__=='__main__':masks();reuse552()
