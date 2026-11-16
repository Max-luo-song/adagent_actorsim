"""GPU history-only TV validation, reusing measured framework-render inputs.

Recorded masks/poses remain an explicit limitation; no closed-loop claim.
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
import json
import os
import sys
import time
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from causal_buffer import CausalWeatherBuffer
from window_contract import validate_manifest,CAMERAS
from apply_weather_window import flow_metrics
HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1]/'framework_repo'


def load_triplet(folder,rows):
    images=[];masks=[]
    for r in rows:
        for kind in ('rgb','mask'):
            if hashlib.sha256((folder/r[kind]).read_bytes()).hexdigest()!=r[kind+'_sha256']:
                raise ValueError(f'{kind} hash changed')
        with Image.open(folder/r['rgb']) as im:
            images.append(np.array(im.convert('RGB').resize((384,256),Image.Resampling.BICUBIC)))
        with Image.open(folder/r['mask']) as im:
            masks.append(np.array(im.convert('L').resize((384,256),Image.Resampling.NEAREST)))
    return np.stack(images),np.stack(masks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.inputs=args.inputs.resolve();args.output=args.output.resolve()
    data=_hugsim_expand(json.loads((args.inputs/'manifest.json').read_text()));records=validate_manifest(data)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(formal_evaluation_admitted=False,diagnostic_uncertified=True,
        gpu_inference_verified=False,causal_structure_verified=False,
        future_independence_replay_verified=False,weather_closed_loop_verified=False,
        temporal_consistency_verified=False,cross_view_geometry_consistency_verified=False,
        observation_source=data['observation_source'],reused_inputs=str(args.inputs),
        input_manifest_sha256=hashlib.sha256((args.inputs/'manifest.json').read_bytes()).hexdigest(),
        mask_scope=data['mask_source'],recorded_views_only=True,
        no_new_scene_render=True,training=False,weather='rainy',frames=data['frames'],
        source_code_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__).resolve(),HERE/'causal_buffer.py',
            FRAMEWORK/'WeatherEdit/background_editing/src/cyclegan_turbo.py',
            FRAMEWORK/'WeatherEdit/background_editing/src/my_utils/cross_att.py')},
        windows=[],input_size=[384,256],temporal_view_attention=True,
        emitted_outputs='current last triplet only; prior outputs immutable')
    try:
        sys.path.insert(0,str(FRAMEWORK/'WeatherEdit/background_editing/src'));os.chdir(FRAMEWORK)
        from cyclegan_turbo import CycleGAN_Turbo
        report['packages']={n:md.version(n) for n in ('torch','diffusers','transformers','peft','huggingface-hub')}
        began=time.monotonic()
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=CycleGAN_Turbo(3,3,weather_type='rainy',with_seg=True,tv_att=True,
            pretrained_path=str(FRAMEWORK/'WeatherEdit/background_editing/ckpts/with_seg.pkl'))
        model.eval();model.requires_grad_(False)
        rng_state=torch.get_rng_state();cuda_rng_state=torch.cuda.get_rng_state_all()
        report['rng_policy']='seed42 model initialization; reset to post-initialization RNG state for each window/replay'
        call_times=[]
        def predict(images,masks):
            torch.set_rng_state(rng_state);torch.cuda.set_rng_state_all(cuda_rng_state)
            x=torch.from_numpy(images).permute(0,3,1,2).float().cuda()/127.5-1
            labels=torch.from_numpy(masks)[:,None].float().cuda()/255
            start=time.monotonic()
            with torch.inference_mode():
                output=model(x,labels,0,9,mode='inference',direction='a2b',
                    caption='Picture of a rainy weather scene',with_seg=True)
            array=output.detach().float().cpu().numpy().transpose(0,2,3,1)
            if array.shape!=(9,256,384,3) or not np.isfinite(array).all():raise RuntimeError('invalid inference output')
            call_times.append(time.monotonic()-start)
            return np.clip((array*.5+.5)*255,0,255).astype('uint8')
        buffer=CausalWeatherBuffer();clear=[];weather=[]
        for j,frame in enumerate(data['frames']):
            # Read current pixels only here; future image files have not been loaded.
            rows=records[j*3:j*3+3];images,masks=load_triplet(args.inputs,rows)
            if j==0:first_images,first_masks=images.copy(),masks.copy()
            result,audit=buffer.process(frame,rows[0]['timestamp_s'],'scene552:unedited',images,masks,predict)
            report['windows'].append(audit);clear.append(images);weather.append(result)
            for c in range(3):
                Image.fromarray(images[c]).save(args.output/f'{frame:03d}_{c}_before.png')
                Image.fromarray(result[c]).save(args.output/f'{frame:03d}_{c}_weather.png')
            pair=Image.new('RGB',(384*3,256*2))
            for c in range(3):
                pair.paste(Image.fromarray(images[c]),(c*384,0));pair.paste(Image.fromarray(result[c]),(c*384,256))
            pair.save(args.output/f'{frame:03d}_causal_three_camera.png')
        # Reset only the causal buffer, reuse the loaded model AFTER it saw f82.
        # Any prefix difference can reveal model-state leakage or nondeterminism.
        replay,_=CausalWeatherBuffer().process(80,8.,'scene552:unedited',first_images,first_masks,predict)
        difference=np.abs(replay.astype(int)-weather[0].astype(int))
        report.update(gpu_inference_verified=True,
            causal_structure_verified=all(max(w['window_timestamps'])<=w['current_timestamp'] for w in report['windows']),
            future_independence_replay_verified=bool(difference.max()==0),
            prefix_replay_max_abs_lsb=int(difference.max()),prefix_replay_mean_abs_lsb=float(difference.mean()),
            elapsed_seconds=time.monotonic()-began,model_calls=len(call_times),call_seconds=call_times)
        report['temporal_flow_diagnostics']={name:[dict(frames=data['frames'][j:j+2],
            **flow_metrics(clear[j][c],clear[j+1][c],weather[j][c],weather[j+1][c]))
            for j in (0,1)] for c,name in enumerate(CAMERAS)}
        offline=args.inputs.parent/'weather'
        if (offline/'report.json').exists():
            report['offline_noncausal_reference']=str(offline/'report.json')
            report['causal_vs_noncausal_pixel_mae']={}
            for j,frame in enumerate(data['frames']):
                for c,name in enumerate(CAMERAS):
                    with Image.open(offline/f'{frame:03d}_{c}_weather.png') as im: reference=np.array(im)
                    report['causal_vs_noncausal_pixel_mae'][f'{frame}:{name}']=float(np.abs(reference.astype(float)-weather[j][c]).mean())
        report['passed_engineering_gate']=(report['causal_structure_verified'] and report['future_independence_replay_verified'])
        if not report['passed_engineering_gate']:raise RuntimeError('causal engineering gate failed; inspect report')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:(args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
