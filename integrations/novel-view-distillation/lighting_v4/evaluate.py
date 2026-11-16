"""Direct ID vs same-camera interpolation/mean, with original real GT only."""

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
import argparse
import sys
import json
import numpy as np
import torch
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE),str(BASE/'quality_v2'),str(BASE/'conditioning_v3')]
from contracts import P26,read,write,sha
import pipeline as v1
from experiment import load_plan,pins as quality_pins
from framework_v2 import Renderer
from check_ids import FRAMES,replace_image_id
from policy import MODES,build_tables,fallback_coefficients,override_lighting

SOURCE=P26/'novel_distillation_v2/runs/20261109T022852Z'
ID_AB=P26/'novel_distillation_v3/id_ab/20261109T040209Z/report.json'


def source_data(source=SOURCE,id_ab=ID_AB):
    source=Path(source).resolve();old=read(id_ab)
    _,plan,rows=load_plan(source)
    if not old.get('completed') or old.get('old_images_reproduced_within_quantization') is not True:raise ValueError('Verified ID A/B result required')
    samples=[r for r in rows if r['kind']=='original' and r['accepted'] and r['source_frame'] in FRAMES]
    cached={(r['frame'],r['cam']):r for r in old['rows']}
    if len(samples)!=27 or set(cached)!={(r['source_frame'],r['camera_id']) for r in samples}:raise ValueError('Source observations differ from ID A/B')
    for r in cached.values():v1.verify_asset(r['old_image']);v1.verify_asset(r['new_image'])
    return plan,samples,cached


def implementation():
    result=quality_pins()
    result.update({str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')})
    p=BASE/'conditioning_v3/check_ids.py';result[str(p)]=sha(p)
    return result


def check():
    plan,samples,cached=source_data()
    return dict(passed=True,samples=len(samples),source_training_samples=sum(not r['source_dataset_test_frame'] for r in cached.values()),
        source_heldout_samples=sum(r['source_dataset_test_frame'] for r in cached.values()),
        expected_actual_gpu_renders=39,modes=list(MODES),source_checkpoint=plan['assets']['source_checkpoint'],
        code=implementation(),gpu_executed=False)


def run(output,source=SOURCE,id_ab=ID_AB):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    plan,samples,cached=source_data(source,id_ab)
    out.mkdir(parents=True)
    write(out/'inputs.json',dict(schema='source_holdout_lighting_v4',source_plan=v1.asset(Path(source)/'plan.json'),
        id_ab=v1.asset(id_ab),checkpoint=plan['assets']['source_checkpoint'],code=implementation(),
        actual_gpu_renders_expected=39,source_frame_clock_changed=False,weights_updated=False))
    renderer=Renderer(plan['parent_job']['inputs']['scene_manifest']['path'],plan['parent_job']['task']['renderer_bundle'])
    results=[];panels={};render_count=0
    try:
        model=renderer.trainer.models['GlobalLighting'];cameras=renderer.backend.cameras_data
        train_frames=tuple(int(x) for x in renderer.backend.dataset.train_timesteps)
        test_frames=set(int(x) for x in renderer.backend.dataset.test_timesteps)
        if set(train_frames)&test_frames:raise ValueError('Training and source holdout overlap')
        denominator=max(1,renderer.trainer.num_full_images-1)
        tables=build_tables(model,cameras,train_frames,denominator)
        write(out/'lighting_sources.json',{str(cam):dict(frames=list(t['frames']),image_ids=list(t['image_ids']),time_denominator=denominator) for cam,t in tables.items()})
        for row in samples:
            frame=row['source_frame'];cam=row['camera_id'];heldout=frame in test_frames
            if heldout!=cached[frame,cam]['source_dataset_test_frame']:raise ValueError('Source split changed')
            with torch.no_grad():
                data=renderer.frame(row['ego_to_world'],frame,cam)
                index=replace_image_id(data,cameras[cam],frame,model.appearance_embedding.num_embeddings)
                camera_pose=data['cam_infos']['camera_to_world'].clone();clock=data['image_infos']['normed_time'].clone()
                target=torch.tensor(v1.image(row['target']['path']),device=model.appearance_embedding.weight.device,dtype=torch.float32)/255.
                mask=torch.tensor(v1.image(row['valid_mask']['path'])[...,0]>0,device=target.device)
                scores={};images={};audits={};executed={}
                for mode in MODES:
                    coefficients,audit=fallback_coefficients(tables,mode,frame,cam);audits[mode]=audit
                    if not heldout and mode!='direct':
                        if coefficients is not None:raise ValueError('Fallback altered a training frame')
                        scores[mode]=dict(scores['direct']);images[mode]=images['direct'];executed[mode]=False
                        continue
                    with override_lighting(model,coefficients):image=renderer.rgb(data)['rgb'].clamp(0,1)
                    if not torch.equal(camera_pose,data['cam_infos']['camera_to_world']) or not torch.equal(clock,data['image_infos']['normed_time']):raise ValueError('Geometry/clock changed')
                    error=image[mask]-target[mask]
                    scores[mode]=dict(l1=float(error.abs().mean()),psnr=float(-10*error.square().mean().clamp_min(1e-12).log10()))
                    images[mode]=v1.save_image(out/f'{frame:03d}_{cam}_{mode}.png',(image.cpu().numpy()*255).round().astype('uint8'))
                    render_count+=1;executed[mode]=True
                reference=cached[frame,cam]['training_id_result']
                results.append(dict(frame=frame,cam=cam,training_image_id=index,source_heldout=heldout,
                    old_id=cached[frame,cam]['old'],scores=scores,images=images,policy_sources=audits,render_executed=executed,
                    direct_reproduction_psnr_delta_db=scores['direct']['psnr']-reference['psnr'],
                    direct_reproduction_l1_delta=scores['direct']['l1']-reference['l1']))
            panels.setdefault(frame,{})[cam]=([images[m]['path'] for m in MODES]+[row['target']['path']])
            write(out/'progress.json',dict(completed_samples=len(results),expected_samples=27,actual_renders=render_count,latest=f'{frame}/{cam}'))
            print(f'Lighting comparison {len(results)}/27: frame={frame} cam={cam} heldout={heldout}',flush=True)
        summary={}
        for subset in ['all','source_train','source_holdout']:
            selected=[r for r in results if subset=='all' or r['source_heldout']==(subset=='source_holdout')]
            summary[subset]={}
            for mode in ('old_id',)+MODES:
                values=[r['old_id'] if mode=='old_id' else r['scores'][mode] for r in selected]
                summary[subset][mode]=dict(samples=len(selected),
                    l1=float(np.mean([v['l1'] for v in values])),psnr=float(np.mean([v['psnr'] for v in values])))
        reproduction_ok=all(abs(r['direct_reproduction_psnr_delta_db'])<.001 and abs(r['direct_reproduction_l1_delta'])<1e-5 for r in results)
        old=summary['source_holdout']['old_id'];direct=summary['source_holdout']['direct'];eligible=[];gates={}
        for mode in MODES[1:]:
            score=summary['source_holdout'][mode]
            gates[mode]=dict(source_train_unchanged=all(r['scores'][mode]==r['scores']['direct'] for r in results if not r['source_heldout']),
                source_holdout_l1_not_worse_than_old=score['l1']<=old['l1'],
                source_holdout_psnr_not_worse_than_old=score['psnr']>=old['psnr'],
                source_holdout_better_than_direct=score['l1']<direct['l1'],direct_reproduced=reproduction_ok)
            if all(gates[mode].values()):eligible.append(mode)
        candidate=min(eligible,key=lambda m:summary['source_holdout'][m]['l1']) if eligible else None
        for frame,cams in panels.items():
            sheet=Image.new('RGB',(1440,1240),'#161616');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'Frozen model frame {frame} | DIRECT / SAME-CAMERA INTERPOLATION / SAME-CAMERA MEAN / REAL GT',fill='white')
            for col,cam in enumerate([1,0,2]):
                for line,path in enumerate(cams[cam]):
                    with Image.open(path) as image:
                        image=image.convert('RGB');image.thumbnail((480,280));sheet.paste(image,(480*col,40+300*line))
            sheet.save(out/f'comparison_{frame:03d}.png')
        report=dict(completed=True,summary=summary,rows=results,quality_checks=gates,
            recommended_for_data_regeneration=candidate,direct_baseline_reproduced=reproduction_ok,
            actual_gpu_renders=render_count,source_train_outputs_reused=True,weights_updated=False,
            geometry_or_actor_clock_changed=False,automatic_promotion=False,gpu_executed=True,
            caveat='6 source-holdout images from 2 timestamps only; no unseen-trajectory or temporal quality certification.')
        write(out/'report.json',report)
        if candidate is not None:
            write(out/'conditioning_policy.json',dict(schema='scene_appearance_conditioning_v1',
                checkpoint=plan['assets']['source_checkpoint'],training_image_id='camera.unique_img_idx[source_frame]',
                source_training_frame_policy='direct',source_holdout_policy=candidate,
                field='GlobalLighting output SH coefficients',time_denominator=denominator,
                source_train_frames=list(train_frames),validation_report=v1.asset(out/'report.json'),
                approved_for='next data-regeneration experiment only',runtime_backend_replaced=False,
                temporal_or_unseen_view_certified=False))
    finally:renderer.close()
    return {k:v for k,v in report.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--output')
    a=p.parse_args()
    if not a.check and not a.output:p.error('--output required')
    print(json.dumps(check() if a.check else run(a.output),indent=2))
