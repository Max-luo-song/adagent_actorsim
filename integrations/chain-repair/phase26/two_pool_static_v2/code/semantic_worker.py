import argparse,time,os
import numpy as np
from PIL import Image
from common import *
def save_png(path,array):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');Image.fromarray(array).save(tmp,format='PNG');os.replace(tmp,path)
def main(scene,limit=None,device='cpu'):
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 source=read(ASSETS/str(scene)/'source_manifest.json');folder=ASSETS/str(scene)/'semantic19';folder.mkdir(exist_ok=True);rows=[];began=time.monotonic()
 from portable_segformer import PortableSegformer
 print('LOADING semantic model CPU',scene,flush=True);model=PortableSegformer(device);audit=model.audit();audit.update(device=device,torch_threads=4,torch_version=torch.__version__,adapter=asset(LOCAL/'portable_segformer.py'));contract=__import__('hashlib').sha256(__import__('json').dumps(audit,sort_keys=True).encode()).hexdigest()
 for frame in source['sampled_frames']:
  for cam in range(5):
   name='%03d_%d'%(frame,cam);receipt=folder/(name+'.json');img=source['frames_data'][frame]['camera_files'][str(cam)+':image'];verify(img)
   if receipt.exists():
    row=read(receipt)
    if row['inference_contract_sha256']!=contract or row['image_sha256']!=img['sha256']:raise ValueError('Semantic cache mismatch')
    for key in ('file','confidence_file','margin_file'):verify({'path':str(folder/row[key]),'sha256':row['sha256' if key=='file' else key.replace('_file','_sha256')]})
   else:
    start=time.monotonic();mask,confidence,margin=model.predict(img['path'],return_confidence=True)
    row={'frame':frame,'camera':cam,'image_sha256':img['sha256'],'inference_contract_sha256':contract,'inference_seconds':time.monotonic()-start}
    for key,ar in [('file',mask),('confidence_file',confidence),('margin_file',margin)]:
     target=folder/(name+{'file':'','confidence_file':'_confidence','margin_file':'_margin'}[key]+'.png');save_png(target,ar);row[key]=target.name;row['sha256' if key=='file' else key.replace('_file','_sha256')]=sha(target)
    write(receipt,row)
   rows.append(row);write(folder/'progress.json',{'scene':scene,'completed':len(rows),'total':len(source['sampled_frames'])*5,'elapsed_seconds':time.monotonic()-began,'last_frame':frame,'last_camera':cam,'pid':os.getpid(),'device':device,'patch_validation':False});print('SEM',scene,len(rows),'/',len(source['sampled_frames'])*5,'last_seconds',round(row['inference_seconds'],2),flush=True)
   if limit is not None and len(rows)>=limit:return
   if device=='cpu' and (OUT/'CPU_YIELD_TO_GPU').exists():raise SystemExit(75)
 write(folder/'semantic_manifest.json',{'schema':'map_sampled_original_semantics_v1','scene':scene,'coverage':'all5 original cameras for every preregistered map sample; not an all-time-image export','sampled_frames':source['sampled_frames'],'maps':rows,'inference_audit':audit,'inference_contract_sha256':contract,'source_manifest':asset(ASSETS/str(scene)/'source_manifest.json'),'model_predictions_not_truth':True,'training_performed':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--limit',type=int);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();main(a.scene,a.limit,a.device)
