"""One recorded image + its real mask through archived WeatherEdit, no training.

Single-frame/single-view smoke only; not temporal consistency or closed-loop
weather admission. Source loader uses relative paths, so run from framework_repo.
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

import argparse
import hashlib
import importlib.metadata as md
import time
import json
import os
import sys
pass  # Path is provided by the portable boundary above
import numpy as np

HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1]/'framework_repo'
DATA=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training/552'))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,default=DATA/'images/080_0.jpg')
    ap.add_argument('--mask',type=Path,default=DATA/'dynamic_masks/all/080_0.png')
    ap.add_argument('--weather',choices=('rainy','snowy','foggy'),default='rainy')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.image=args.image.resolve();args.mask=args.mask.resolve();args.output=args.output.resolve()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from PIL import Image
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    resources=FRAMEWORK/'WeatherEdit/background_editing/ckpts'
    verification=_hugsim_expand(json.loads((resources/'download_verification.json').read_text()))
    if not verification.get('resource_structure_ready') or not verification.get('weather_sha256_verified'):
        raise RuntimeError('verified resources required')
    before=Image.open(args.image).convert('RGB');mask=Image.open(args.mask).convert('L')
    if before.size!=mask.size: raise ValueError('source image/mask size mismatch')
    # Use the real segmentation mask, no all-zero or invented substitute.
    before=before.resize((512,512),Image.Resampling.BICUBIC)
    mask=mask.resize((512,512),Image.Resampling.NEAREST)
    sys.path.insert(0,str(FRAMEWORK/'WeatherEdit/background_editing/src'))
    os.chdir(FRAMEWORK)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(diagnostic_uncertified=True,formal_evaluation_admitted=False,
                gpu_inference_verified=False,weather=args.weather,
                image=str(args.image),mask=str(args.mask),image_sha256=sha(args.image),
                mask_sha256=sha(args.mask),download_verification=verification,
                scope='single recorded image; no multiview/temporal/closed-loop weather validation',
                training=False,depth_modified=False,collision_geometry_modified=False,
                packages={n:md.version(n) for n in ('torch','diffusers','transformers','peft','huggingface-hub')},
                source_code_sha256={str(p):sha(p) for p in (
                    FRAMEWORK/'WeatherEdit/background_editing/src/cyclegan_turbo.py',
                    FRAMEWORK/'WeatherEdit/background_editing/src/model.py',Path(__file__).resolve())})
    try:
        from cyclegan_turbo import CycleGAN_Turbo
        began=time.monotonic()
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        x=torch.from_numpy(np.array(before).copy()).permute(2,0,1).float()[None].cuda()/127.5-1
        label=torch.from_numpy(np.array(mask).copy()).float()[None,None].cuda()/255
        model=CycleGAN_Turbo(1,1,weather_type=args.weather,with_seg=True,tv_att=False,
                             pretrained_path=str(resources/'with_seg.pkl'))
        model.eval();model.requires_grad_(False)
        with torch.inference_mode():
            result=model(x,label,0,1,mode='inference',direction='a2b',
                         caption=f'Picture of a {args.weather} weather scene',with_seg=True)
        array=result[0].detach().float().cpu().numpy().transpose(1,2,0)
        if array.shape!=(512,512,3) or not np.isfinite(array).all(): raise RuntimeError('invalid RGB output')
        array=np.clip((array*.5+.5)*255,0,255).astype('uint8')
        difference=np.abs(array.astype(float)-np.array(before).astype(float))
        before.save(args.output/'before.png');mask.save(args.output/'source_mask.png')
        comparison=Image.new('RGB',(1024,512));comparison.paste(before,(0,0))
        comparison.paste(Image.fromarray(array),(512,0));comparison.save(args.output/'before_vs_weather.png')
        Image.fromarray(array).save(args.output/'weather.png')
        Image.fromarray((difference.max(-1)>3).astype('uint8')*255).save(args.output/'changed_rgb_mask.png')
        report.update(gpu_inference_verified=True,elapsed_seconds=time.monotonic()-began,
                      mean_abs_rgb_difference=float(difference.mean()),
                      weather_quality_manually_reviewed=False)
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
