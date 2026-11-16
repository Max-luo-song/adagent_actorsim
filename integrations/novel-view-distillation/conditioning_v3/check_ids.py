"""Read-only original-view A/B: frozen weights/rays, old vs training image IDs."""

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

pass  # Path is provided by the portable boundary above
import sys
import argparse
import json
import numpy as np
import torch
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE),str(BASE/'quality_v2')]
from contracts import P26,read,write,sha
import pipeline as v1
from experiment import load_plan
from framework_v2 import Renderer

SOURCE=P26/'novel_distillation_v2/runs/20261109T022852Z'
FRAMES=[0,9,15,24,30,39,45,54,60]


def training_image_id(camera,frame,image_count):
    value=camera.unique_img_idx[frame]
    value=int(value.item()) if hasattr(value,'item') else int(value)
    if not 0<=value<image_count:raise ValueError('Training appearance ID outside checkpoint embedding')
    return value


def replace_image_id(data,camera,frame,image_count):
    value=training_image_id(camera,frame,image_count)
    old=data['image_infos']['img_idx']
    # Preserve normalized_time, rays, pose, frame counter and GlobalLighting's
    # original training-time denominator. Change only the image-embedding ID.
    data['image_infos']['img_idx']=torch.full_like(old,value,dtype=torch.long)
    return value


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    _,plan,rows=load_plan(source)
    samples=[r for r in rows if r['kind']=='original' and r['accepted'] and r['source_frame'] in FRAMES]
    if len(samples)!=len(FRAMES)*3:raise ValueError('Expected 27 paired source observations')
    out.mkdir(parents=True)
    write(out/'inputs.json',dict(source_plan=v1.asset(source/'plan.json'),
        original_checkpoint=plan['assets']['source_checkpoint'],
        original_job=plan['assets']['job.json'],script=v1.asset(__file__),
        helpers={str(p):sha(p) for p in [BASE/'quality_v2/framework_v2.py',BASE/'contracts.py',BASE/'pipeline.py',BASE/'quality_v2/experiment.py']},
        changed='Only image_infos.img_idx, using camera.unique_img_idx[frame]',weights_updated=False))
    renderer=Renderer(plan['parent_job']['inputs']['scene_manifest']['path'],plan['parent_job']['task']['renderer_bundle'])
    results=[];panels={}
    try:
        n=renderer.trainer.models['GlobalLighting'].appearance_embedding.num_embeddings
        for row in samples:
            frame=row['source_frame'];cam=row['camera_id']
            with torch.no_grad():
                data=renderer.frame(row['ego_to_world'],frame,cam)
                previous_id=int(data['image_infos']['img_idx'].flatten()[0]);pose=data['cam_infos']['camera_to_world'].clone()
                clock=data['image_infos']['normed_time'].clone()
                old=renderer.rgb(data)['rgb'].clamp(0,1)
                index=replace_image_id(data,renderer.backend.cameras_data[cam],frame,n)
                if not torch.equal(clock,data['image_infos']['normed_time']) or not torch.equal(pose,data['cam_infos']['camera_to_world']):
                    raise RuntimeError('Conditioning A/B changed geometry or time')
                new=renderer.rgb(data)['rgb'].clamp(0,1)
                target=torch.tensor(v1.image(row['target']['path']),device=new.device,dtype=new.dtype)/255.
                mask=torch.tensor(v1.image(row['valid_mask']['path'])[...,0]>0,device=new.device)
                baseline=torch.tensor(v1.image(row['before']['path']),device=new.device,dtype=new.dtype)/255.
                def measure(image):
                    error=image[mask]-target[mask]
                    return dict(l1=float(error.abs().mean()),psnr=float(-10*error.square().mean().clamp_min(1e-12).log10()))
                key=f'{frame:03d}_{cam}'
                a=v1.save_image(out/f'{key}_old.png',(old.cpu().numpy()*255).round().astype('uint8'))
                b=v1.save_image(out/f'{key}_training_id.png',(new.cpu().numpy()*255).round().astype('uint8'))
                results.append(dict(frame=frame,cam=cam,old_id=previous_id,training_id=index,
                    source_dataset_test_frame=frame in set(renderer.backend.dataset.test_timesteps),
                    old=measure(old),training_id_result=measure(new),
                    old_reproduction_l1=float((old-baseline).abs().mean()),old_image=a,new_image=b))
            panels.setdefault(frame,{})[cam]=(a['path'],b['path'],row['target']['path'])
            write(out/'progress.json',dict(completed_samples=len(results),expected_samples=27,latest=key))
            print(f'Image-ID A/B {len(results)}/27: frame={frame} camera={cam} id={previous_id}->{index}',flush=True)
        summaries=[]
        for subset in ['all','source_train_frames','source_test_frames']:
            selected=[r for r in results if subset=='all' or r['source_dataset_test_frame']==(subset=='source_test_frames')]
            summaries.append(dict(subset=subset,samples=len(selected),
                old_l1=float(np.mean([r['old']['l1'] for r in selected])),new_l1=float(np.mean([r['training_id_result']['l1'] for r in selected])),
                old_psnr=float(np.mean([r['old']['psnr'] for r in selected])),new_psnr=float(np.mean([r['training_id_result']['psnr'] for r in selected]))))
        for frame,cameras in panels.items():
            sheet=Image.new('RGB',(1440,1000),'#161616');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'Original-view frozen-checkpoint A/B | frame={frame} | OLD ID / TRAINING ID / REAL GT',fill='white')
            for column,cam in enumerate([1,0,2]):
                for line,path in enumerate(cameras[cam]):
                    with Image.open(path) as image:
                        image=image.convert('RGB');image.thumbnail((480,280));sheet.paste(image,(column*480,40+line*300))
            sheet.save(out/f'comparison_{frame:03d}.png')
        report=dict(completed=True,summary=summaries,rows=results,weights_updated=False,
            old_images_reproduced_within_quantization=all(r['old_reproduction_l1']<=.0021 for r in results),
            geometry_and_time_changed=False,gpu_executed=True,
            note='Original real GT A/B only. Source hold-out frames reported separately; no retraining or pseudo-GT regeneration.')
        write(out/'report.json',report)
    finally:renderer.close()
    return {k:v for k,v in report.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default=str(SOURCE));p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.output),indent=2))
