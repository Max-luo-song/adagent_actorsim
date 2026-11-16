"""Resumable source-image semantic export. No simulator or GPU required."""

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

import argparse,json,os,time,hashlib,platform
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from scene_data import CLASSES,sha,write_json

def atomic_json(path,payload):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');write_json(tmp,payload);tmp.replace(path)
def png(path,array):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');Image.fromarray(array).save(tmp,format='PNG');tmp.replace(path)
def compare(pred,path):
 old=np.asarray(Image.open(path))>0;union=np.sum(pred|old)
 return {'iou':float(np.sum(pred&old)/union) if union else 1.,'disagree_pixels':int(np.sum(pred!=old))}
def export(bundle,device='cpu'):
 from portable_segformer import PortableSegformer
 import torch
 bundle=Path(bundle);meta=_hugsim_expand(json.loads((bundle/'source_manifest.json').read_text()));out=bundle/'semantic19';out.mkdir(exist_ok=True);start=time.monotonic();model=PortableSegformer(device);audit=model.audit();audit.update(adapter_sha256=sha(Path(__file__).with_name('portable_segformer.py')),torch_version=torch.__version__,torch_threads=torch.get_num_threads(),device=device);digest=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest();rows=[]
 source={'schema':'full19_semantics_v2','classes':CLASSES,'source_checkpoint_sha256':meta['source_checkpoint_sha256'],'model_sha256':audit['model_sha256'],'inference_audit':audit,'inference_contract_sha256':digest,'semantic_authority':'model predictions, not manually annotated truth','probabilities_calibrated':False,'confidence_encoding':'uint8 round(255*p_top1), margin=round(255*(p_top1-p_top2))','maps':rows}
 for frame in meta['frames_data']:
  for cam in range(5):
   name='%03d_%d'%(frame['frame'],cam);src=frame['camera_files'][str(cam)+':image'];path=Path(src['path']);target=out/(name+'.png');record=out/(name+'.json');confidence=out/(name+'_confidence.png');margin=out/(name+'_margin.png')
   if sha(path)!=src['sha256']:raise ValueError('Changed source image')
   if record.exists():
    row=_hugsim_expand(json.loads(record.read_text()))
    if row['inference_contract_sha256']!=digest or row['image_sha256']!=src['sha256']:raise ValueError('Cannot resume a different inference contract')
    for field,artifact in [('sha256',target),('confidence_sha256',confidence),('margin_sha256',margin)]:
     if sha(artifact)!=row[field]:raise ValueError('Changed semantic result')
   else:
    began=time.monotonic();mask,conf,mg=model.predict(path,return_confidence=True)
    if mask.shape!=tuple(frame['image_size_wh'][str(cam)][::-1]) or not np.isin(mask,range(19)).all():raise ValueError('Invalid semantic map')
    png(target,mask);png(confidence,conf);png(margin,mg)
    row={'frame':frame['frame'],'camera':cam,'file':target.name,'sha256':sha(target),'confidence_file':confidence.name,'confidence_sha256':sha(confidence),'margin_file':margin.name,'margin_sha256':sha(margin),'image_sha256':src['sha256'],'inference_contract_sha256':digest,'inference_seconds':time.monotonic()-began,'road_comparison':compare(mask==0,frame['camera_files'][str(cam)+':road']['path']),'sky_comparison':compare(mask==10,frame['camera_files'][str(cam)+':sky']['path'])};atomic_json(record,row)
   rows.append(row);elapsed=time.monotonic()-start;atomic_json(out/'progress.json',{'stage':'semantic19_export','completed':len(rows),'total':meta['frames']*5,'elapsed_seconds':elapsed,'last_frame':frame['frame'],'last_camera':cam,'pid':os.getpid(),'new_closed_loop_executed':False});print('Semantic maps',len(rows),'/',meta['frames']*5,'frame',frame['frame'],'cam',cam,flush=True)
 if len(rows)!=995:raise ValueError('Incomplete full semantic coverage')
 atomic_json(out/'semantic_manifest.json',source);print('Full semantic export complete',flush=True);return source

def main():
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args();export(a.bundle,a.device)
if __name__=='__main__':main()
