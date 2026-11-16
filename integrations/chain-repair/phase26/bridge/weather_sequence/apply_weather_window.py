"""Nine-image TV-attention weather test; explicitly offline/noncausal."""

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

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
import time
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from window_contract import validate_manifest,CAMERAS
HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1]/'framework_repo'


def flow_metrics(a,b,wa,wb):
    import cv2
    g0=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);g1=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
    f=cv2.calcOpticalFlowFarneback(g0,g1,None,.5,3,15,3,5,1.2,0)
    backward=cv2.calcOpticalFlowFarneback(g1,g0,None,.5,3,15,3,5,1.2,0)
    h,w=g0.shape;xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    mx,my=xx+f[...,0],yy+f[...,1]
    back=cv2.remap(backward,mx,my,cv2.INTER_LINEAR)
    valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)&(np.linalg.norm(f+back,axis=-1)<1.5)
    if not valid.any():return dict(valid_fraction=0,clear_aligned_mae=None,weather_aligned_mae=None)
    bwarp=cv2.remap(b,mx,my,cv2.INTER_LINEAR);wwarp=cv2.remap(wb,mx,my,cv2.INTER_LINEAR)
    return dict(valid_fraction=float(valid.mean()),
                clear_aligned_mae=float(np.abs(a.astype(float)-bwarp.astype(float))[valid].mean()),
                weather_aligned_mae=float(np.abs(wa.astype(float)-wwarp.astype(float))[valid].mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.inputs=args.inputs.resolve();args.output=args.output.resolve()
    data=_hugsim_expand(json.loads((args.inputs/'manifest.json').read_text()));records=validate_manifest(data)
    arrays=[];masks=[]
    for r in records:
        rgb_path=args.inputs/r['rgb']
        if hashlib.sha256(rgb_path.read_bytes()).hexdigest()!=r['rgb_sha256']:raise ValueError('input RGB changed')
        if hashlib.sha256((args.inputs/r['mask']).read_bytes()).hexdigest()!=r['mask_sha256']:raise ValueError('input mask changed')
        with Image.open(rgb_path) as im: a=im.convert('RGB').resize((384,256),Image.Resampling.BICUBIC)
        with Image.open(args.inputs/r['mask']) as im: mask=im.convert('L').resize((384,256),Image.Resampling.NEAREST)
        arrays.append(np.asarray(a).copy());masks.append(np.asarray(mask).copy())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(formal_evaluation_admitted=False,diagnostic_uncertified=True,
        gpu_inference_verified=False,temporal_consistency_verified=False,
        cross_view_geometry_consistency_verified=False,weather_closed_loop_verified=False,
        observation_source=data['observation_source'],weather='rainy',frames=data['frames'],
        batch_order='frame-major then CAM_FRONT/CAM_FRONT_LEFT/CAM_FRONT_RIGHT',
        input_manifest_sha256=hashlib.sha256((args.inputs/'manifest.json').read_bytes()).hexdigest(),
        input_size=[384,256],temporal_view_attention=True,offline_noncausal=True,
        future_window_frames_used=True,training=False,
        mask_scope=data['mask_source'],manual_review_required=True,
        metrics_scope='optical-flow photometric diagnostics and per-view correction means; no automatic consistency certification')
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        sys.path.insert(0,str(FRAMEWORK/'WeatherEdit/background_editing/src'));os.chdir(FRAMEWORK)
        from cyclegan_turbo import CycleGAN_Turbo
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        report['packages']={n:md.version(n) for n in ('torch','diffusers','transformers','peft','huggingface-hub')}
        began=time.monotonic()
        x=torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).float().cuda()/127.5-1
        labels=torch.from_numpy(np.stack(masks))[:,None].float().cuda()/255
        model=CycleGAN_Turbo(3,3,weather_type='rainy',with_seg=True,tv_att=True,
                            pretrained_path=str(FRAMEWORK/'WeatherEdit/background_editing/ckpts/with_seg.pkl'))
        model.eval();model.requires_grad_(False)
        with torch.inference_mode():
            output=model(x,labels,0,9,mode='inference',direction='a2b',
                         caption='Picture of a rainy weather scene',with_seg=True)
        value=output.detach().float().cpu().numpy().transpose(0,2,3,1)
        if value.shape!=(9,256,384,3) or not np.isfinite(value).all():raise RuntimeError('invalid output')
        weather=np.clip((value*.5+.5)*255,0,255).astype('uint8')
        report.update(gpu_inference_verified=True,elapsed_seconds=time.monotonic()-began)
        report['images']=[]
        for i,(r,a,b) in enumerate(zip(records,arrays,weather)):
            stem=f"{r['frame']:03d}_{r['camera_id']}"
            Image.fromarray(a).save(args.output/f'{stem}_before.png');Image.fromarray(b).save(args.output/f'{stem}_weather.png')
            correction=b.astype(float)-a.astype(float)
            report['images'].append(dict(frame=r['frame'],camera=r['camera'],
                mean_abs_difference=float(np.abs(correction).mean()),mean_rgb_correction=correction.mean((0,1)).tolist()))
        for frame_index,frame in enumerate(data['frames']):
            sheet=Image.new('RGB',(384*3,256*2))
            for cam_index in range(3):
                idx=frame_index*3+cam_index
                sheet.paste(Image.fromarray(arrays[idx]),(cam_index*384,0))
                sheet.paste(Image.fromarray(weather[idx]),(cam_index*384,256))
            sheet.save(args.output/f'{frame:03d}_three_camera_before_after.png')
        report['temporal_flow_diagnostics']={}
        for cam_index,name in enumerate(CAMERAS):
            report['temporal_flow_diagnostics'][name]=[
                dict(frames=data['frames'][j:j+2],**flow_metrics(arrays[j*3+cam_index],arrays[(j+1)*3+cam_index],
                    weather[j*3+cam_index],weather[(j+1)*3+cam_index])) for j in (0,1)]
        report['success_scope']='9 framework-rendered inputs processed by one TV-attention window; consistency requires review'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:(args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
