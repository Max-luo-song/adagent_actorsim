"""Fresh supervision and distillation with verified image-ID/lighting conditioning."""

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
import hashlib
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE),str(BASE/'quality_v2'),str(BASE/'conditioning_v3'),str(BASE/'lighting_v4')]
from contracts import P26,read,write,sha
import pipeline as legacy
import framework
from experiment import PROFILES,pins as quality_pins
import experiment as quality
from check_ids import replace_image_id
from policy import build_tables,fallback_coefficients,override_lighting

BaseRenderer=framework.Renderer
POLICY=P26/'novel_distillation_v4/lighting_ab/20261109T050625Z/conditioning_policy.json'
TASK=HERE/'task.json'
CONTEXT=None
RENDER_AUDIT=[]
SOURCE_MAPS=[]


def family_pins():
    result=quality_pins()
    result.update({str(p):sha(p) for p in BASE.iterdir() if p.suffix in ('.py','.sh')})
    result.update({str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')})
    for p in [BASE/'conditioning_v3/check_ids.py',BASE/'lighting_v4/policy.py']:result[str(p)]=sha(p)
    return result


def lighting_fingerprint(model):
    h=hashlib.sha256()
    for key,t in sorted(model.state_dict().items()):
        h.update(json.dumps([key,str(t.dtype),list(t.shape)]).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class ConditionedRenderer(BaseRenderer):
    def __init__(self,*args,**kwargs):
        if CONTEXT is None:raise RuntimeError('Conditioning context was not configured')
        super().__init__(*args,**kwargs)
        self.light=self.trainer.models['GlobalLighting']
        if lighting_fingerprint(self.light)!=CONTEXT['global_lighting_fingerprint']:
            raise ValueError('Candidate changed the frozen GlobalLighting weights')
        self.conditioning=CONTEXT['policy']
        trained=tuple(int(f) for f in self.backend.dataset.train_timesteps)
        if trained!=tuple(self.conditioning['source_train_frames']):raise ValueError('Source reconstruction split changed')
        denominator=max(1,self.trainer.num_full_images-1)
        if denominator!=self.conditioning['time_denominator']:raise ValueError('Lighting training clock changed')
        self.tables=build_tables(self.light,self.backend.cameras_data,trained,denominator)
        self.source_maps={str(cam):camera.unique_img_idx.detach().cpu().tolist() for cam,camera in self.backend.cameras_data.items()}
        SOURCE_MAPS.append(dict(camera_image_ids=self.source_maps,source_train_frames=list(trained),time_denominator=denominator,
            policy_sha256=CONTEXT['policy_asset']['sha256'],global_lighting_fingerprint=CONTEXT['global_lighting_fingerprint']))

    def frame(self,ego_pose,frame,cam):
        data=super().frame(ego_pose,frame,cam)
        index=replace_image_id(data,self.backend.cameras_data[cam],frame,self.light.appearance_embedding.num_embeddings)
        _,audit=fallback_coefficients(self.tables,self.conditioning['source_holdout_policy'],frame,cam)
        data['_appearance_conditioning']=dict(schema='render_appearance_conditioning_v1',source_frame=frame,camera_id=cam,
            image_id=index,lighting=audit,policy_sha256=CONTEXT['policy_asset']['sha256'],
            global_lighting_fingerprint=CONTEXT['global_lighting_fingerprint'])
        return data

    def rgb(self,data,layers=False):
        audit=data['_appearance_conditioning']
        coefficient,_=fallback_coefficients(self.tables,self.conditioning['source_holdout_policy'],audit['source_frame'],audit['camera_id'])
        with override_lighting(self.light,coefficient):result=super().rgb(data,layers=layers)
        RENDER_AUDIT.append(audit)
        return result


def validate_policy(path):
    asset=legacy.asset(path);p=read(path)
    if p.get('schema')!='scene_appearance_conditioning_v1' or p.get('source_training_frame_policy')!='direct' or p.get('source_holdout_policy')!='interpolate':
        raise ValueError('Expected the verified direct/interpolate conditioning policy')
    legacy.verify_asset(p['checkpoint']);legacy.verify_asset(p['validation_report'])
    report=read(p['validation_report']['path'])
    if report.get('recommended_for_data_regeneration')!='interpolate' or not all(report['quality_checks']['interpolate'].values()):
        raise ValueError('Conditioning policy lacks GPU A/B acceptance')
    return asset,p


def prepare(task,output,policy_path=POLICY):
    policy_asset,policy=validate_policy(policy_path)
    legacy.implementation=family_pins
    result=legacy.prepare(task,output)
    out=Path(output).resolve();job=read(out/'job.json')
    if job['inputs']['checkpoint']!=policy['checkpoint']:raise ValueError('Conditioning policy belongs to another scene checkpoint')
    state=torch.load(job['inputs']['checkpoint']['path'],map_location='cpu',weights_only=False)
    # Fingerprint only the frozen model; state_dict has identical names/tensor values.
    from types import SimpleNamespace
    fingerprint=lighting_fingerprint(SimpleNamespace(state_dict=lambda:state['models']['GlobalLighting']))
    del state
    job['inputs']['conditioning_policy']=policy_asset
    job['inputs']['conditioning_validation']=policy['validation_report']
    job['adapter_schema']='conditioned_scene_distillation_v5'
    job['conditioning']=dict(policy_asset=policy_asset,policy=policy,global_lighting_fingerprint=fingerprint)
    job['training_profile']=dict(name='structure',policy=PROFILES['structure'])
    job['execution_adapter']='scoped ConditionedRenderer; frozen legacy data stages and quality_v2 training'
    write(out/'job.json',job)
    write(out/'plan.json',dict(schema='conditioned_scene_training_v5',parent=str(out),job=legacy.asset(out/'job.json'),code=family_pins(),
        profiles={'structure':PROFILES['structure']},parent_job=job,
        assets={'source_checkpoint':job['inputs']['checkpoint'],'source_config':job['inputs']['config']},
        reused_old_pseudo_gt=False,appearance_conditioning=job['conditioning'],geometry_frozen=True))
    result.update(adapter='conditioned_scene_distillation_v5',old_pseudo_gt_reused=False,
        source_holdout_policy='interpolate',expected_training_steps_before_target_filter=2*result['samples_by_split']['train']*PROFILES['structure']['epochs'])
    return result


def configure(output):
    global CONTEXT
    legacy.implementation=family_pins
    out,job=legacy.job_at(output)
    if job.get('adapter_schema')!='conditioned_scene_distillation_v5':raise ValueError('Refusing an old-conditioning job')
    asset,p=validate_policy(job['conditioning']['policy_asset']['path'])
    if asset!=job['conditioning']['policy_asset'] or p!=job['conditioning']['policy']:raise ValueError('Conditioning policy changed')
    plan=read(out/'plan.json')
    legacy.verify_asset(plan['job'])
    if plan['code']!=family_pins() or plan['profiles']!={'structure':PROFILES['structure']}:raise ValueError('Training plan changed')
    CONTEXT=job['conditioning']
    framework.Renderer=ConditionedRenderer
    quality.Renderer=ConditionedRenderer
    return out,job,plan


def enrich_render_manifest(out,job):
    path=out/'render/manifest.json';manifest=read(path)
    if not manifest.get('completed') or not SOURCE_MAPS:raise ValueError('Completed conditioned renders required')
    source=SOURCE_MAPS[-1]
    audit_map={}
    for audit in RENDER_AUDIT:
        key=(audit['source_frame'],audit['camera_id'])
        if key in audit_map and audit_map[key]!=audit:raise ValueError('Conditioning changed between recorded/new poses')
        audit_map[key]=audit
    for row in manifest['rows']:
        key=(row['source_frame'],row['camera_id'])
        if key not in audit_map:raise ValueError('Rendered row has no actual conditioning receipt')
        row['appearance_conditioning']=audit_map[key]
        if row['appearance_conditioning']['image_id']!=source['camera_image_ids'][str(row['camera_id'])][row['source_frame']]:raise ValueError('Wrong image ID')
        write(out/'render/rows'/f'{row["id"]}.json',row)
    write(out/'render/conditioning_sources.json',source)
    manifest['appearance_conditioning_sources']=legacy.asset(out/'render/conditioning_sources.json')
    manifest['conditioning_policy']=job['conditioning']['policy_asset']
    write(path,manifest)


def conditioned_rows(out,job):
    rows=legacy.target_rows(out)
    manifest=read(out/'render/manifest.json')
    legacy.verify_asset(manifest['appearance_conditioning_sources'])
    if manifest['conditioning_policy']!=job['conditioning']['policy_asset']:raise ValueError('Render policy mismatch')
    source=read(manifest['appearance_conditioning_sources']['path'])
    by_id={r['id']:r['appearance_conditioning'] for r in manifest['rows']}
    for row in rows:
        a=row.get('appearance_conditioning')
        if a!=by_id.get(row['id']) or a is None:raise ValueError('Mixed/old pseudo-GT conditioning')
        if a['policy_sha256']!=job['conditioning']['policy_asset']['sha256'] or a['global_lighting_fingerprint']!=job['conditioning']['global_lighting_fingerprint']:raise ValueError('Frozen conditioning state mismatch')
        if a['image_id']!=source['camera_image_ids'][str(row['camera_id'])][row['source_frame']]:raise ValueError('Invalid image ID in supervision')
    return rows


def dispatch(stage,output):
    RENDER_AUDIT.clear();SOURCE_MAPS.clear()
    out,job,plan=configure(output)
    if stage=='render':
        result=legacy.render_pack(out);enrich_render_manifest(out,job);return result
    if stage=='pseudo-gt':
        if 'appearance_conditioning_sources' not in read(out/'render/manifest.json'):raise ValueError('Old render pack rejected')
        result=legacy.pseudo_gt(out);conditioned_rows(out,job);return result
    def scoped_plan(path):
        if Path(path).resolve()!=out:raise ValueError('Wrong conditioned experiment output')
        configure(out)
        return out,plan,conditioned_rows(out,job)
    quality.load_plan=scoped_plan
    if stage=='train':
        result=quality.train(out,'structure')
        receipt=read(out/'structure/train/receipt.json')
        write(out/'conditioned_candidate.json',dict(schema='conditioned_scene_candidate_v5',
            checkpoint=receipt['checkpoint'],scene_manifest=receipt['scene_manifest'],
            conditioning_policy=job['conditioning']['policy_asset'],
            global_lighting_fingerprint=job['conditioning']['global_lighting_fingerprint'],
            source_job=legacy.asset(out/'job.json'),training_plan=legacy.asset(out/'plan.json'),
            requires_conditioned_renderer=True,automatic_promotion=False))
        return result
    if stage=='evaluate':
        result=quality.evaluate(out,'structure')
        path=out/'structure/evaluation/report.json';report=read(path)
        report.update(data_conditioning_policy=job['conditioning']['policy_asset'],
            fresh_pseudo_gt=True,old_v2_absolute_pseudo_gt_metrics_comparable=False)
        write(path,report)
        return result
    raise ValueError('Unknown stage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','render','pseudo-gt','train','evaluate'])
    parser.add_argument('--task',default=str(TASK));parser.add_argument('--policy',default=str(POLICY));parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.task,args.output,args.policy) if args.stage=='prepare' else dispatch(args.stage,args.output),indent=2,allow_nan=False))
