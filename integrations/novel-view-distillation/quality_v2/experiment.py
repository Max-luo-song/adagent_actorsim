"""Reuse verified v1 supervision for independent quality ablations; no new DiFix run."""

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
pass  # Path is provided by the portable boundary above
import json
import shutil
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from contracts import ROOT,P26,read,write,sha,merge_checkpoint,check_checkpoint_invariants
import pipeline as v1
from framework_v2 import Renderer,trainer_tensor_placements
from quality_policy import PROFILES,ADAM_EPS,SEED,select_parameters,epoch_schedule,loss_terms,objective,clamp_color_update,parameter_stats

PARENT=P26/'novel_distillation_v1/runs/20261108T023505Z'


def pins():
    paths=[p for p in HERE.iterdir() if p.suffix in ('.py','.sh')]
    paths += [HERE.parent/'contracts.py',HERE.parent/'pipeline.py']
    return {str(p):sha(p) for p in paths}


def prepare(output,parent=PARENT,profiles=None):
    out=Path(output).resolve();parent=Path(parent).resolve()
    if out.exists():raise FileExistsError(out)
    profiles=list(PROFILES) if profiles is None else profiles
    if not profiles or len(set(profiles))!=len(profiles) or set(profiles)-PROFILES.keys():raise ValueError('Unknown/duplicate profiles')
    job=read(parent/'job.json');receipt=read(parent/'distill/receipt.json')
    if not receipt.get('completed'):raise ValueError('Completed parent training required')
    for value in job['inputs'].values():v1.verify_asset(value)
    v1.verify_asset(receipt['candidate']);v1.verify_asset(receipt['pseudo_manifest'])
    rows=v1.target_rows(parent)
    if {r['edit_revision'] for r in rows}!={0}:raise ValueError('Unedited scene required')
    if sha(parent/'compiled_trajectory.json')!=job['trajectory_sha256']:raise ValueError('Trajectory changed')
    if not (parent/'evaluation/report.json').is_file():raise ValueError('Completed v1 evaluation required for comparison')
    source=read(job['inputs']['source_report']['path'])
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(job['inputs']['config']['path'])
    data=Path(source['data_dir']);metadata={}
    used={r['source_frame'] for r in rows}
    for frame in used|{0}:
        f=data/'ego_pose'/f'{source["start_timestep"]+frame:03d}.txt';metadata[str(f)]=sha(f)
    for cam in [0,1,2]:
        for folder in ['extrinsics','intrinsics']:
            f=data/folder/f'{cam}.txt';metadata[str(f)]=sha(f)
    assets={k:v1.asset(parent/k) for k in ['job.json','compiled_trajectory.json','render/manifest.json','pseudo_gt/manifest.json','distill/receipt.json','evaluation/report.json']}
    assets['source_checkpoint']=job['inputs']['checkpoint'];assets['source_config']=job['inputs']['config']
    assets['renderer_manifest']=job['inputs']['renderer_manifest']
    out.mkdir(parents=True)
    write(out/'plan.json',dict(schema='novel_quality_ablation_v2',parent=str(parent),parent_job=job,
        assets=assets,source_metadata=metadata,profiles={p:PROFILES[p] for p in profiles},code=pins(),
        initialized_from='original scene checkpoint for every profile',reused_pseudo_gt=True,
        geometry_frozen=True,actors_frozen=True,cameras_frozen=True,teacher_regenerated=False,
        seed=SEED,adam_eps=ADAM_EPS,dc_trust_radius=.25,sh_rest_trust_radius=.1,trust_region_profiles=['balanced','structure'],
        color_reference_lr=float(cfg.trainer.gaussian_optim_general_cfg.sh_dc.lr),
        evaluation_role='inherited v1 held-out frames used as validation; no unseen-trajectory certification'))
    return dict(prepared=True,output=str(out),profiles=list(profiles),
        expected_steps={p:PROFILES[p]['epochs']*(3*sum(r['accepted'] and r['split']=='train' and r['kind']=='novel' for r in rows)
            if PROFILES[p]['schedule']=='v1_blocks' else 2*max(sum(r['accepted'] and r['split']=='train' and r['kind']==k for r in rows) for k in ['novel','original'])) for p in profiles},gpu_executed=False)


def load_plan(output):
    out=Path(output).resolve();plan=read(out/'plan.json')
    if plan['code']!=pins():raise ValueError('Frozen experiment code changed')
    for p,profile in plan['profiles'].items():
        if profile!=PROFILES[p]:raise ValueError('Profile changed')
    verified={}
    for asset in list(plan['assets'].values())+list(plan['parent_job']['inputs'].values()):
        if asset['path'] in verified:
            if verified[asset['path']]!=asset['sha256']:raise ValueError('Conflicting source hashes')
            continue
        v1.verify_asset(asset);verified[asset['path']]=asset['sha256']
    for path,expected in plan['source_metadata'].items():
        if sha(path)!=expected:raise ValueError('Source calibration/clock changed: '+path)
    rows=v1.target_rows(Path(plan['parent']))
    if any(r['checkpoint_sha256']!=plan['assets']['source_checkpoint']['sha256'] for r in rows):raise ValueError('Supervision checkpoint mismatch')
    return out,plan,rows


def _tensor_rgb(row,renderer):
    data=renderer.frame(row['ego_to_world'],row['source_frame'],row['camera_id'])
    result=renderer.rgb(data)['rgb'].clamp(0,1)
    target=torch.as_tensor(v1.image(row['target']['path']),device=result.device,dtype=result.dtype)/255.
    mask=torch.as_tensor(v1.image(row['valid_mask']['path'])[...,0]>0,device=result.device)
    return result,target,mask


def _save_candidate(out,plan,selected):
    path=out/'checkpoint_distilled.pth'
    base=torch.load(plan['assets']['source_checkpoint']['path'],map_location='cpu',weights_only=False)
    torch.save(merge_checkpoint(base,selected),path)
    del base
    old=torch.load(plan['assets']['source_checkpoint']['path'],map_location='cpu',weights_only=False)
    new=torch.load(path,map_location='cpu',weights_only=False)
    changed=check_checkpoint_invariants(old,new)
    del old,new
    shutil.copy2(plan['assets']['source_config']['path'],out/'config.yaml')
    parent_job=plan['parent_job'];source=read(parent_job['inputs']['source_report']['path'])
    source.update(checkpoint=str(path),source_config=str(out/'config.yaml'));write(out/'source_report.json',source)
    scene=read(parent_job['inputs']['scene_manifest']['path']);scene['source_report']=str(out/'source_report.json')
    vehicle=Path(scene['vehicle_config'])
    if not vehicle.is_absolute():vehicle=Path(parent_job['inputs']['scene_manifest']['path']).parent/vehicle
    scene['vehicle_config']=str(vehicle.resolve());write(out/'scene_manifest.json',scene)
    return path,changed


def train(output,name):
    out,plan,rows=load_plan(output)
    if name not in plan['profiles']:raise ValueError('Profile not admitted')
    stage=out/name/'train';stage.mkdir(parents=True,exist_ok=False)
    profile=plan['profiles'][name];groups={k:[r for r in rows if r['accepted'] and r['split']=='train' and r['kind']==k] for k in ['novel','original']}
    renderer=Renderer(plan['parent_job']['inputs']['scene_manifest']['path'],plan['parent_job']['task']['renderer_bundle'])
    try:
        selected,optimizer_groups=select_parameters(renderer.trainer,profile)
        initial={key:p.detach().clone() for key,p in selected.items()}
        optimizer=torch.optim.Adam(optimizer_groups,eps=ADAM_EPS)
        rng=np.random.default_rng(SEED);step=0;trained=set();active={'.'.join(k):0 for k in selected}
        write(stage/'parameters.json',dict(active=[dict(name=g['name'],lr=g['lr'],numel=g['params'][0].numel()) for g in optimizer_groups],
            excluded_background_sh_rest=bool(getattr(renderer.trainer.models['Background'],'use_pbr',False))))
        inventory=trainer_tensor_placements(renderer.trainer)
        write(stage/'device_inventory.json',{k:dict(device=str(t.device),shape=list(t.shape)) for k,t in inventory.items()})
        with (stage/'steps.jsonl').open('w',buffering=1) as stream:
            for epoch in range(profile['epochs']):
                epoch_losses=[]
                for row in epoch_schedule(groups,profile,rng):
                    if row['split']!='train':raise ValueError('Validation row entered training')
                    pred,target,mask=_tensor_rgb(row,renderer)
                    terms=loss_terms(pred,target,mask);loss=objective(terms,profile)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    grads=[p.grad for p in selected.values() if p.grad is not None]
                    if not grads or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError('Missing/nonfinite color gradients')
                    for key,p in selected.items():
                        if p.grad is not None and float(p.grad.detach().abs().max())>0:active['.'.join(key)]+=1
                    optimizer.step()
                    if profile['trust_region']:clamp_color_update(selected,initial)
                    step+=1;trained.add(row['id'])
                    record=dict(step=step,epoch=epoch,sample=row['id'],kind=row['kind'],split='train',
                        loss=float(loss.detach()),l1=float(terms['l1'].detach()),ssim=float(terms['ssim'].detach()),edge=float(terms['edge'].detach()),
                        ssim_windows=terms['ssim_windows'])
                    if step==1 or step%25==0:record['parameters']=parameter_stats(selected,initial,optimizer)
                    stream.write(json.dumps(record,allow_nan=False)+'\n');epoch_losses.append(record['loss'])
                write(stage/'progress.json',dict(epoch=epoch+1,epochs=profile['epochs'],steps=step,mean_loss=float(np.mean(epoch_losses)),active_gradient_steps=active))
                print(f'{name}: epoch {epoch+1}/{profile["epochs"]}, updates={step}, loss={np.mean(epoch_losses):.6f}',flush=True)
                if any(count==0 for count in active.values()):raise RuntimeError('An admitted color group received no nonzero gradient')
        candidate,changed=_save_candidate(stage,plan,selected)
        write(stage/'receipt.json',dict(completed=True,profile=name,policy=profile,steps=step,
            checkpoint=v1.asset(candidate),scene_manifest=v1.asset(stage/'scene_manifest.json'),
            plan=v1.asset(out/'plan.json'),changed_weights=changed,geometry_and_actors_unchanged=True,
            initialized_from=plan['assets']['source_checkpoint'],trained_sample_ids=sorted(trained),
            active_gradient_steps=active,final_parameters=parameter_stats(selected,initial,optimizer),
            automatic_promotion=False,gpu_executed=True))
    finally:renderer.close()
    return dict(completed=True,profile=name,steps=step,checkpoint=str(candidate))


def evaluate(output,name):
    out,plan,rows=load_plan(output);trained=out/name/'train';receipt=read(trained/'receipt.json')
    for k in ['checkpoint','scene_manifest','plan']:v1.verify_asset(receipt[k])
    stage=out/name/'evaluation';stage.mkdir(exist_ok=False)
    renderer=Renderer(trained/'scene_manifest.json',plan['parent_job']['task']['renderer_bundle'])
    scores=[];panels=[]
    try:
        for row in rows:
            if row['split']!='eval' or not row['accepted']:continue
            if row['id'] in receipt['trained_sample_ids']:raise ValueError('Train/validation leakage')
            with torch.no_grad():
                pred,target,mask=_tensor_rgb(row,renderer)
                before=torch.as_tensor(v1.image(row['before']['path']),device=pred.device,dtype=pred.dtype)/255.
                def measure(rgb):
                    terms=loss_terms(rgb,target,mask);mse=(rgb-target).square()[mask].mean()
                    return dict(l1=float(terms['l1']),ssim_loss=float(terms['ssim']),edge_l1=float(terms['edge']),psnr=float(-10*torch.log10(mse.clamp_min(1e-12))))
                before_score=measure(before);after_score=measure(pred)
                saved=v1.save_image(stage/f'{row["id"]}_after.png',(pred.cpu().numpy()*255).round().astype('uint8'))
            scores.append(dict(id=row['id'],kind=row['kind'],before=before_score,after=after_score))
            if row['kind']=='novel':panels.append((row,saved['path']))
            write(stage/'progress.json',dict(completed_samples=len(scores),latest=row['id']))
        averages={kind:{when:{key:float(np.mean([r[when][key] for r in scores if r['kind']==kind]))
            for key in ['l1','ssim_loss','edge_l1','psnr']} for when in ['before','after']} for kind in ['novel','original']}
        novel=averages['novel'];original=averages['original']
        l1_gain=1-novel['after']['l1']/novel['before']['l1'];edge_gain=1-novel['after']['edge_l1']/max(novel['before']['edge_l1'],1e-12)
        drop=original['before']['psnr']-original['after']['psnr']
        criteria=dict(novel_l1_improved_at_least_5pct=l1_gain>=.05,novel_edge_error_improved_at_least_2pct=edge_gain>=.02,
            novel_ssim_improved=novel['after']['ssim_loss']<novel['before']['ssim_loss'],original_psnr_drop_at_most_05db=drop<=.5)
        v1.make_comparison(panels,stage,read(Path(plan['parent'])/'compiled_trajectory.json')['fps']/5.)
        report=dict(completed=True,profile=name,averages=averages,metrics=scores,
            novel_l1_gain_fraction=l1_gain,novel_edge_gain_fraction=edge_gain,original_psnr_drop_db=drop,
            quality_checks=criteria,quality_screen_passed=all(criteria.values()),
            comparison=str(stage/'before_teacher_after.mp4'),checkpoint=receipt['checkpoint'],
            geometry_changed=False,temporal_or_unseen_view_certified=False,closed_loop_validation_required=True,
            automatic_promotion=False,gpu_executed=True)
        write(stage/'report.json',report)
    finally:renderer.close()
    return {k:v for k,v in report.items() if k not in ['averages','metrics']}


def compare(output):
    out,plan,_=load_plan(output);parent=read(Path(plan['parent'])/'evaluation/report.json')
    reports=[]
    for name in plan['profiles']:
        r=read(out/name/'evaluation/report.json')
        reports.append(dict(profile=name,quality_screen_passed=r['quality_screen_passed'],
            l1_gain_fraction=r['novel_l1_gain_fraction'],edge_gain_fraction=r['novel_edge_gain_fraction'],
            original_psnr_drop_db=r['original_psnr_drop_db'],novel_after=r['averages']['novel']['after'],video=r['comparison']))
    eligible=[r for r in reports if r['quality_screen_passed']]
    best=min(eligible,key=lambda r:r['novel_after']['l1'])['profile'] if eligible else None
    result=dict(completed=True,v1_novel_l1=parent['averages']['novel']['after']['l1'],profiles=reports,
        recommended_for_further_review=best,automatic_promotion=False,
        note='Validation comparisons only. Higher teacher similarity does not certify scene fidelity.')
    write(out/'comparison.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','evaluate','compare','check'])
    p.add_argument('--output');p.add_argument('--parent',default=str(PARENT));p.add_argument('--profile',choices=list(PROFILES));p.add_argument('--profiles',nargs='+',choices=list(PROFILES))
    args=p.parse_args()
    if args.stage=='check':
        print(json.dumps(dict(cuda_available=torch.cuda.is_available(),profiles=PROFILES),indent=2));return
    if not args.output:p.error('--output required')
    if args.stage in ['train','evaluate'] and not args.profile:p.error('--profile required')
    actions={'prepare':lambda:prepare(args.output,args.parent,args.profiles),'train':lambda:train(args.output,args.profile),
        'evaluate':lambda:evaluate(args.output,args.profile),'compare':lambda:compare(args.output)}
    print(json.dumps(actions[args.stage](),indent=2,allow_nan=False))


if __name__=='__main__':main()
