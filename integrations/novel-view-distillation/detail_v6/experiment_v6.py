"""Independent warm-start edge/footprint ablations; never promote runtime weights."""

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
import copy
import json
import shutil
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'regeneration_v5'),str(HERE.parent)]
import workflow as w5
from contracts import P26,read,write,sha
from quality_policy import loss_terms,epoch_schedule,SEED
import detail_policy as policy
v1=w5.legacy
DEFAULT_PARENT=P26/'novel_distillation_v5/runs/20261109T054653Z'


def pins():
    return {str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')}


def prepare(output,parent=DEFAULT_PARENT):
    out=Path(output).resolve();parent,job,_=w5.configure(parent)
    rows=w5.conditioned_rows(parent,job)
    receipt=read(parent/'structure/train/receipt.json')
    if not receipt['completed']:raise ValueError('Completed warm start required')
    assets={k:v1.asset(parent/k) for k in ['job.json','plan.json','render/manifest.json','pseudo_gt/manifest.json',
        'structure/train/receipt.json','structure/evaluation/report.json','compiled_trajectory.json']}
    assets.update(warm_checkpoint=receipt['checkpoint'],warm_scene=receipt['scene_manifest'],source_config=job['inputs']['config'])
    for a in assets.values():v1.verify_asset(a)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    write(out/'plan.json',dict(schema='supported_background_detail_v6',parent=str(parent),code=pins(),
        family_code=w5.family_pins(),assets=assets,profiles=policy.PROFILES,seed=SEED,
        reused_corrected_v5_supervision=True,teacher_regenerated=False,automatic_promotion=False,
        depth_reference='warm v5 expected camera-Z render; no geometric ground truth',
        support='Background centers visible in at least two distinct original TRAIN timestamps',
        footprint_bounds=dict(scale_ratio=[.85,1.05],opacity_logit_delta=.2),
        frozen=['all centers','all rotations','RoadNodes geometry','actors','lighting','calibration']))
    return dict(output=str(out),prepared=True,rows=len(rows),profiles=list(policy.PROFILES),gpu_executed=False)


def load(output):
    out=Path(output).resolve();plan=read(out/'plan.json')
    if plan['schema']!='supported_background_detail_v6' or plan['code']!=pins() or plan['profiles']!=policy.PROFILES:
        raise ValueError('Frozen v6 implementation/profile changed')
    parent,job,_=w5.configure(plan['parent'])
    if plan['family_code']!=w5.family_pins():raise ValueError('Parent implementation changed')
    for a in plan['assets'].values():v1.verify_asset(a)
    return out,plan,job,w5.conditioned_rows(parent,job)


def renderer(plan,job,scene=None):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; use the GPU terminal')
    return w5.ConditionedRenderer(scene or plan['assets']['warm_scene']['path'],job['task']['renderer_bundle'])


def render_row(r,row):
    data=r.frame(row['ego_to_world'],row['source_frame'],row['camera_id']);result=r.rgb(data)
    pred=result['rgb'].clamp(0,1)
    target=torch.as_tensor(v1.image(row['target']['path']),device=pred.device,dtype=pred.dtype)/255.
    mask=torch.as_tensor(v1.image(row['valid_mask']['path'])[...,0]>0,device=pred.device)
    return result,pred,target,mask


def metrics(pred,target,mask):
    terms=loss_terms(pred,target,mask)
    return dict(l1=float(terms['l1']),ssim_loss=float(terms['ssim']),edge_l1=float(terms['edge']),
        psnr=float(-10*torch.log10((pred-target).square()[mask].mean().clamp_min(1e-12))))


def averages(scores,field):
    return {kind:{key:float(np.mean([s[field][key] for s in scores if s['kind']==kind]))
        for key in scores[0][field]} for kind in ['novel','original']}


def cache(output):
    out,plan,job,rows=load(output);stage=out/'reference';stage.mkdir(exist_ok=False)
    r=renderer(plan,job);records={};scores=[]
    try:
        r.trainer.requires_grad_(False)
        for m in r.trainer.models.values():m.requires_grad_(False)
        means=r.trainer.models['Background']._means.detach().cpu().double().numpy()
        support=policy.SupportAccumulator(len(means))
        # Original training rows alone admit footprint points; holdout views cannot admit points.
        for row in sorted([x for x in rows if x['accepted']],key=lambda x:(x['source_frame'],x['camera_id'],x['kind'])):
            with torch.no_grad():
                result,pred,target,mask=render_row(r,row)
                depth=result['depth'].squeeze(-1);alpha=result['opacity'].squeeze(-1)
                valid=mask&torch.isfinite(depth)&torch.isfinite(alpha)&(depth>.2)&(depth<120)&(alpha>=.8)
                if not bool(valid.any()):raise ValueError('Empty static depth support: '+row['id'])
                ref=dict(depth=depth.cpu(),opacity=alpha.cpu(),mask=valid.cpu())
                path=stage/(row['id']+'.pt');torch.save(ref,path)
                record=dict(reference=v1.asset(path))
                if row['split']=='eval':
                    record['rgb']=v1.save_image(stage/(row['id']+'.png'),(pred.cpu().numpy()*255).round().astype('uint8'))
                    score=metrics(pred,target,mask);record['score']=score
                    scores.append(dict(kind=row['kind'],score=score))
            if row['kind']=='original' and row['split']=='train':
                support.add(row['source_frame'],policy.visible_centers(means,row['camera_to_world'],row['K'],ref))
            records[row['id']]=record
            print('v6 reference: '+row['id'],flush=True)
        admitted=support.finish();torch.save(torch.from_numpy(admitted),stage/'support.pt')
        avg=averages(scores,'score');old=read(plan['assets']['structure/evaluation/report.json']['path'])['averages']
        checks={kind:abs(avg[kind]['l1']-old[kind]['after']['l1'])<1e-5 for kind in avg}
        if not all(checks.values()):raise ValueError('Warm-start rendering did not reproduce v5 metrics')
        write(stage/'manifest.json',dict(completed=True,plan=v1.asset(out/'plan.json'),rows=records,
            support=v1.asset(stage/'support.pt'),supported_points=int(admitted.sum()),total_background_points=len(means),
            min_distinct_source_train_timestamps=2,baseline=avg,v5_reproduced=checks,
            geometry_ground_truth=False,gpu_executed=True))
    finally:r.close()
    return dict(completed=True,supported_points=int(admitted.sum()),baseline=avg)


def references(out):
    manifest=read(out/'reference/manifest.json')
    if not manifest['completed']:raise ValueError('Incomplete references')
    v1.verify_asset(manifest['plan']);v1.verify_asset(manifest['support'])
    return manifest,torch.load(manifest['support']['path'],map_location='cpu',weights_only=True)


def reference_row(manifest,row,device):
    asset=manifest['rows'][row['id']]['reference'];v1.verify_asset(asset)
    return {k:t.to(device) for k,t in torch.load(asset['path'],map_location='cpu',weights_only=True).items()}


def probe(output):
    out,plan,job,rows=load(output);refs,support=references(out);r=renderer(plan,job);records=[]
    try:
        selected,_=policy.select_parameters(r.trainer,policy.PROFILES['footprint']);support=support.to(next(iter(selected.values())).device)
        chosen=[x for x in rows if x['accepted'] and x['kind']=='novel' and x['split']=='train' and x['source_frame'] in [15,45]]
        if not chosen:raise ValueError('No designated gradient probe samples')
        active={'.'.join(k):False for k in selected}
        for row in chosen:
            result,pred,target,mask=render_row(r,row);terms=loss_terms(pred,target,mask);term_grads={};record=dict(id=row['id'],groups={})
            for term in ['l1','ssim','edge']:
                values=torch.autograd.grad(terms[term],list(selected.values()),retain_graph=True,allow_unused=True)
                term_grads[term]=[]
                for (key,p),g in zip(selected.items(),values):
                    g=torch.zeros_like(p) if g is None else g.detach()
                    if key in policy.FOOT_FIELDS:g=g.masked_fill((~support).reshape((-1,)+(1,)*(g.ndim-1)),0)
                    if not torch.isfinite(g).all():raise ValueError('Nonfinite parameter gradient')
                    term_grads[term].append(g)
                    norm=float(torch.linalg.vector_norm(g));name='.'.join(key);active[name]|=norm>0
                    record['groups'].setdefault(name,{})[term+'_raw_norm']=norm
            for i,key in enumerate(selected):
                group=record['groups']['.'.join(key)]
                a=term_grads['edge'][i];b=term_grads['l1'][i]
                group['edge_l1_cosine']=float((a*b).sum()/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)).clamp_min(1e-30))
                group['old_weighted_edge_to_l1']=.1*group['edge_raw_norm']/max(.8*group['l1_raw_norm'],1e-30)
                group['new_weighted_edge_to_l1']=8*group['old_weighted_edge_to_l1']
            records.append(record);print('v6 parameter gradients: '+row['id'],flush=True)
            del result,pred,terms,term_grads,values
        color_ok=all(active['.'.join(k)] for k in policy.COLOR_FIELDS)
        foot_ok=color_ok and int(support.sum())>=100 and all(active['.'.join(k)] for k in policy.FOOT_FIELDS)
        if not color_ok:raise ValueError('An admitted color group has no real render gradient')
        admitted=['edge_color']+(['footprint'] if foot_ok else [])
        report=dict(completed=True,admitted_profiles=admitted,active_gradients=active,records=records,
            supported_points=int(support.sum()),reference=v1.asset(out/'reference/manifest.json'),
            weights_updated=False,gpu_executed=True,footprint_skipped_reason=None if foot_ok else 'Insufficient support/nonzero footprint gradients')
        write(out/'probe.json',report)
    finally:r.close()
    return report


def save_candidate(stage,plan,selected,support):
    before=torch.load(plan['assets']['warm_checkpoint']['path'],map_location='cpu',weights_only=False)
    after=copy.deepcopy(before)
    for (model,field),p in selected.items():after['models'][model][field]=p.detach().cpu().clone()
    after.pop('optimizer',None)
    changed=policy.validate_checkpoint(before,after,set(selected),support.cpu())
    path=stage/'checkpoint_distilled.pth';torch.save(after,path)
    del before,after
    # Verify the serialized artifact, not merely the in-memory proposal.
    before=torch.load(plan['assets']['warm_checkpoint']['path'],map_location='cpu',weights_only=False)
    after=torch.load(path,map_location='cpu',weights_only=False)
    policy.validate_checkpoint(before,after,set(selected),support.cpu())
    del before,after
    shutil.copy2(plan['assets']['source_config']['path'],stage/'config.yaml')
    scene=read(plan['assets']['warm_scene']['path']);source=read(scene['source_report'])
    source.update(checkpoint=str(path),source_config=str(stage/'config.yaml'));write(stage/'source_report.json',source)
    scene['source_report']=str(stage/'source_report.json');write(stage/'scene_manifest.json',scene)
    return path,changed


def train(output,name):
    out,plan,job,rows=load(output);refs,support=references(out);p=policy.PROFILES[name]
    probe_report=read(out/'probe.json');v1.verify_asset(probe_report['reference'])
    if name not in probe_report['admitted_profiles']:raise ValueError('Profile not admitted by actual render gradients')
    stage=out/name/'train';stage.mkdir(parents=True,exist_ok=False);r=renderer(plan,job)
    try:
        selected,groups=policy.select_parameters(r.trainer,p);initial={k:v.detach().clone() for k,v in selected.items()}
        support=support.to(next(iter(selected.values())).device);optimizer=torch.optim.Adam(groups,eps=1e-8)
        data={k:[x for x in rows if x['accepted'] and x['split']=='train' and x['kind']==k] for k in ['novel','original']}
        rng=np.random.default_rng(SEED);trained=set();step=0;active={'.'.join(k):0 for k in selected}
        with (stage/'steps.jsonl').open('w',buffering=1) as stream:
            for epoch in range(p['epochs']):
                for row in epoch_schedule(data,dict(schedule='interleaved'),rng):
                    if row['split']!='train':raise ValueError('Validation data entered training')
                    result,pred,target,mask=render_row(r,row);terms=loss_terms(pred,target,mask)
                    loss=sum(p[k+'_weight']*terms[k] for k in ['l1','ssim','edge'])
                    if p['footprint']:
                        ref=reference_row(refs,row,pred.device);d,a=policy.depth_terms(result['depth'],result['opacity'],ref)
                        loss=loss+.5*d+.1*a+1e-4*policy.footprint_anchor(selected,initial,support)
                    optimizer.zero_grad(set_to_none=True);loss.backward();policy.restrict_gradients(selected,support)
                    for key,param in selected.items():
                        g=param.grad
                        if g is None or not torch.isfinite(g).all():raise ValueError('Missing/nonfinite admitted gradient')
                        active['.'.join(key)]+=int(bool(g.abs().max()>0))
                    optimizer.step();policy.clamp_updates(selected,initial,support)
                    step+=1;trained.add(row['id'])
                    record=dict(step=step,epoch=epoch+1,id=row['id'],loss=float(loss.detach()),
                        **{k:float(terms[k].detach()) for k in ['l1','ssim','edge']})
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                    if step==1 or step%25==0:print(f'v6 {name}: {step}, loss={record["loss"]:.6f}',flush=True)
                if not all(active.values()):raise ValueError('An admitted group never received an active gradient')
                write(stage/'progress.json',dict(epochs=epoch+1,steps=step,active_gradient_steps=active))
        checkpoint,changed=save_candidate(stage,plan,selected,support)
        receipt=dict(completed=True,profile=name,steps=step,checkpoint=v1.asset(checkpoint),
            scene_manifest=v1.asset(stage/'scene_manifest.json'),plan=v1.asset(out/'plan.json'),
            reference=v1.asset(out/'reference/manifest.json'),probe=v1.asset(out/'probe.json'),
            trained_sample_ids=sorted(trained),changed_weights=changed,active_gradient_steps=active,
            initialized_from=plan['assets']['warm_checkpoint'],footprint_changed=p['footprint'],
            all_centers_rotations_actors_road_geometry_unchanged=True,automatic_promotion=False,gpu_executed=True)
        write(stage/'receipt.json',receipt)
    finally:r.close()
    return receipt


def quality_checks(avg,baseline,depth,old):
    novel=avg['novel'];warm=baseline['novel']
    return dict(novel_edge_improved_2pct=novel['edge_l1']<=.98*warm['edge_l1'],
        novel_l1_keeps_v5_gain=novel['l1']<=min(1.01*warm['l1'],.95*old['novel']['before']['l1']),
        novel_ssim_not_regressed=novel['ssim_loss']<=1.001*warm['ssim_loss'],
        original_psnr_drop_at_most_02db=avg['original']['psnr']>=baseline['original']['psnr']-.2,
        original_render_depth_stable=depth['original']['mean_relative_depth']<=.005 and depth['original']['p95_relative_depth']<=.02,
        novel_render_depth_stable=depth['novel']['mean_relative_depth']<=.01 and depth['novel']['p95_relative_depth']<=.03,
        no_static_opacity_holes=all(d['mean_opacity_change']<=.01 and d['new_hole_fraction']<=.001 for d in depth.values()))


def evaluate(output,name):
    out,plan,job,rows=load(output);refs,_=references(out);trained=out/name/'train';receipt=read(trained/'receipt.json')
    for k in ['checkpoint','scene_manifest','plan','reference','probe']:v1.verify_asset(receipt[k])
    stage=out/name/'evaluation';stage.mkdir(exist_ok=False);r=renderer(plan,job,receipt['scene_manifest']['path']);scores=[];panels=[]
    try:
        for row in rows:
            if row['split']!='eval' or not row['accepted']:continue
            if row['id'] in receipt['trained_sample_ids']:raise ValueError('Train/validation leakage')
            with torch.no_grad():
                result,pred,target,mask=render_row(r,row);score=metrics(pred,target,mask)
                ref=reference_row(refs,row,pred.device);depth=policy.depth_metrics(result['depth'],result['opacity'],ref)
                saved=v1.save_image(stage/(row['id']+'_after.png'),(pred.cpu().numpy()*255).round().astype('uint8'))
            scores.append(dict(id=row['id'],kind=row['kind'],score=score,depth=depth))
            if row['kind']=='novel':
                before=refs['rows'][row['id']]['rgb'];v1.verify_asset(before)
                panel=dict(row,before=before);panels.append((panel,saved['path']))
            print('v6 '+name+' evaluation: '+row['id'],flush=True)
        avg=averages(scores,'score');depth=averages(scores,'depth')
        old=read(plan['assets']['structure/evaluation/report.json']['path'])['averages']
        checks=quality_checks(avg,refs['baseline'],depth,old)
        fps=read(plan['assets']['compiled_trajectory.json']['path'])['fps']/5.
        v1.make_comparison(panels,stage,fps)
        report=dict(completed=True,profile=name,averages=avg,warm_baseline=refs['baseline'],depth_proxy=depth,
            metrics=scores,quality_checks=checks,quality_screen_passed=all(checks.values()),
            video=str(stage/'before_teacher_after.mp4'),checkpoint=receipt['checkpoint'],
            footprint_changed=policy.PROFILES[name]['footprint'],geometry_ground_truth=False,
            temporal_or_unseen_view_certified=False,closed_loop_validation_required=True,automatic_promotion=False,gpu_executed=True)
        write(stage/'report.json',report)
    finally:r.close()
    return {k:v for k,v in report.items() if k!='metrics'}


def compare(output):
    out,_,_,_=load(output);probe=read(out/'probe.json');reports=[]
    for name in probe['admitted_profiles']:
        report=read(out/name/'evaluation/report.json')
        if not report['completed']:raise ValueError('Incomplete evaluation')
        reports.append({k:report[k] for k in ['profile','averages','depth_proxy','quality_checks','quality_screen_passed','video']})
    eligible=[r for r in reports if r['quality_screen_passed']]
    result=dict(completed=True,profiles=reports,recommended_for_review=min(eligible,key=lambda x:x['averages']['novel']['edge_l1'])['profile'] if eligible else None,
        automatic_promotion=False,closed_loop_validation_required=True)
    write(out/'comparison.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','check','cache','probe','train','evaluate','compare'])
    parser.add_argument('--output',required=True);parser.add_argument('--parent',default=str(DEFAULT_PARENT));parser.add_argument('--profile',choices=list(policy.PROFILES))
    args=parser.parse_args()
    if args.stage in ['train','evaluate'] and args.profile is None:parser.error('--profile is required')
    actions=dict(prepare=lambda:prepare(args.output,args.parent),check=lambda:dict(verified=True,rows=len(load(args.output)[3]),cuda_available=torch.cuda.is_available()),
        cache=lambda:cache(args.output),probe=lambda:probe(args.output),train=lambda:train(args.output,args.profile),
        evaluate=lambda:evaluate(args.output,args.profile),compare=lambda:compare(args.output))
    print(json.dumps(actions[args.stage](),indent=2,allow_nan=False))
