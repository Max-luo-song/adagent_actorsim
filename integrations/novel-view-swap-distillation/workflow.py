"""Offline distillation with explicit unedited replay and position-swapped views."""

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

import argparse,importlib.util,sys,copy
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'integrations/novel-view-distillation';SWAP=ROOT/'integrations/novel-view-position-swap'
sys.path[:0]=[str(HERE),str(BASE),str(SWAP)]
from contracts import P26,read,write,sha,pose,split_frame
from guards import validate_rows,DeferredSceneState,outside_box_mask,camera_row_header
SOURCE=P26/'novel_view_position_swap_v1/scene551_ego_native3'
OUTPUT=P26/'novel_swap_distillation_v1/scene551_ego_native3'
_provider=None

def provider():
    global _provider
    if _provider is None:
        spec=importlib.util.spec_from_file_location('_swap_distillation_conditioned',BASE/'regeneration_v5/workflow.py');_provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(_provider)
    return _provider
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify_asset(a):
    if sha(a['path'])!=a['sha256']:raise ValueError('Changed input: '+a['path'])
def pins():return {str(p):sha(p) for p in sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))+[SWAP/'preview.py']}

def prepare(output):
    out=Path(output).resolve()
    if out.exists():configure(out);return dict(prepared=True,reused=True,output=str(out))
    source_plan=read(SOURCE/'plan.json');preview=read(SOURCE/'preview/report.json')
    if not preview['completed'] or not preview['both_ego_and_vehicle_positions_changed'] or not preview['original_scene_restored']:raise ValueError('Verified position-swap render required')
    for a in list(source_plan['assets'].values())+list(source_plan['source_input_assets'].values()):verify_asset(a)
    for path,expected in source_plan['code'].items():
        if sha(path)!=expected:raise ValueError('Source swap implementation changed')
    # A new supervision pack, starting from the original scene model.
    task=copy.deepcopy(source_plan['source_task']);task['trajectory']=str(SOURCE/'ego_trajectory.json')
    task_path=HERE/'task.json'
    if task_path.exists() and read(task_path)!=task:raise ValueError('Existing swap task changed')
    if not task_path.exists():write(task_path,task)
    result=provider().prepare(task_path,out)
    job=read(out/'job.json');job.update(edits_applied=True,scene_variant_adapter='position_swap_distillation_v1',
        edited_variant='position_swapped',original_replay_variant='original',
        edit_program=asset(SOURCE/'swap_program.npz'),reference_guard='old target ROI neutralized in teacher reference; projected stale region and current dynamics excluded from loss')
    write(out/'job.json',job)
    plan=read(out/'plan.json');plan.update(job=asset(out/'job.json'),parent_job=job,actor_edit_is_external_to_checkpoint=True)
    write(out/'plan.json',plan)
    write(out/'swap_contract.json',dict(schema='position_swap_distillation_v1',code=pins(),source_plan=asset(SOURCE/'plan.json'),
        source_preview=asset(SOURCE/'preview/report.json'),program=asset(SOURCE/'swap_program.npz'),
        effective_actor_tracks=source_plan['assets']['effective_actor_tracks'],
        job=asset(out/'job.json'),plan=asset(out/'plan.json'),
        requires_external_edit_program=True,default_model_changed=False,automatic_promotion=False,
        smpl_runtime=__import__('smpl_runtime').assets()))
    result.update(edit_program=job['edit_program'],source_preview_reused=True,new_pseudo_GT_required=True)
    return result

def configure(output):
    out=Path(output).resolve();contract=read(out/'swap_contract.json')
    if contract['code']!=pins():raise ValueError('Swap-distillation implementation changed')
    for key in ['source_plan','source_preview','program','effective_actor_tracks','job','plan']:verify_asset(contract[key])
    if 'smpl_runtime' not in contract:raise ValueError('Canonical SMPL runtime not registered; recover the failed initialization first')
    for value in contract['smpl_runtime'].values():verify_asset(value)
    module=provider();module.configure(out);job=read(out/'job.json');plan=read(out/'plan.json')
    if job.get('scene_variant_adapter')!='position_swap_distillation_v1' or job.get('edits_applied') is not True:raise ValueError('Edit-aware job required')
    with np.load(contract['program']['path']) as z:program={k:z[k].copy() for k in z.files}
    from preview import TranslationSwap
    class SwapRenderer(module.ConditionedRenderer):
        def __init__(self,*args,**kwargs):
            from smpl_runtime import canonical_layer
            bundle=args[1] if len(args)>1 else kwargs['bundle']
            module.framework.bootstrap(bundle)
            with canonical_layer(contract['smpl_runtime'],bundle):super().__init__(*args,**kwargs)
            self.edit_state=DeferredSceneState(TranslationSwap(self.backend,program))
        def select_row(self,row):self.edit_state.select(row['source_frame'],row['kind']=='novel')
        def close(self):
            try:
                if hasattr(self,'edit_state'):self.edit_state.close()
            finally:super().close()
    legacy=module.legacy;quality=module.quality
    legacy.validate_rows=lambda rows:validate_rows(rows,contract['program']['sha256'],legacy)
    module.framework.Renderer=SwapRenderer;quality.Renderer=SwapRenderer
    if not hasattr(quality,'_unedited_tensor_rgb'):quality._unedited_tensor_rgb=quality._tensor_rgb
    def tensor_rgb(row,renderer):
        renderer.select_row(row)
        return quality._unedited_tensor_rgb(row,renderer)
    quality._tensor_rgb=tensor_rgb
    def load_plan(path):
        if Path(path).resolve()!=out:raise ValueError('Wrong edited job')
        # Do not call the legacy configure here: it would replace SwapRenderer.
        for key in ['job','plan','program']:verify_asset(contract[key])
        return out,plan,module.conditioned_rows(out,job)
    quality.load_plan=load_plan
    return out,job,plan,module,contract,program,SwapRenderer

def render(output):
    import torch
    from scipy.spatial.transform import Rotation
    out,job,plan,module,contract,program,Renderer=configure(output);legacy=module.legacy
    stage=out/'render';stage.mkdir();(stage/'rows').mkdir()
    with np.load(contract['effective_actor_tracks']['path']) as z:tracks={k:z[k].copy() for k in z.files}
    renderer=Renderer(job['task']['scene_manifest'],job['task']['renderer_bundle']);rows=[]
    # Preserve the actual source mapping before any later row validation can fail.
    write(stage/'conditioning_sources.json',module.SOURCE_MAPS[-1])
    source_preview=read(contract['source_preview']['path']);existing={(r['source_frame'],r['camera_id'],r['variant']):r for r in source_preview['records']}
    try:
        for sample in read(out/'compiled_trajectory.json')['samples']:
            frame=sample['source_frame'];split=split_frame(frame,job['frame_stride'])
            if split=='unused':continue
            size=tracks['RigidNodes__instances_size'][3];center=tracks['RigidNodes__instances_trans'][frame,3]
            rotation=Rotation.from_quat(tracks['RigidNodes__instances_quats'][frame,3][[1,2,3,0]]).as_matrix()
            for cam in [0,1,2]:
                reference,raw_path,original_cam=renderer.original(frame,cam)
                original_ego=original_cam@np.linalg.inv(np.asarray(renderer.backend.cameras_data[cam].cam_to_ego))
                for kind,ego in [('original',original_ego),('novel',sample['ego_to_world'])]:
                    header=camera_row_header(frame,cam,kind,split);key=header['id'];renderer.select_row(dict(kind=kind,source_frame=frame));data=renderer.frame(ego,frame,cam)
                    with torch.no_grad():
                        result=renderer.rgb(data,layers=True);rgb=(result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                        mask=renderer.valid_mask(result,data).cpu().numpy()
                    camera_pose=data['cam_infos']['camera_to_world'].cpu().numpy();K=data['cam_infos']['intrinsics'].cpu().numpy()
                    ref_mask=np.ones(reference.shape[:2],bool);ref=reference.copy()
                    if kind=='novel':
                        ref_mask=outside_box_mask(reference.shape[:2],K,original_cam,center,rotation,size)
                        ref[~ref_mask]=128
                        mask &= outside_box_mask(mask.shape,K,camera_pose,center,rotation,size)
                    row=dict(**header,timestamp_s=sample['timestamp_s'],scene=551,
                        checkpoint_sha256=job['inputs']['checkpoint']['sha256'],
                        edit_revision=int(kind=='novel'),scene_variant='position_swapped' if kind=='novel' else 'original',
                        edit_program_sha256=contract['program']['sha256'],reference_sanitized=kind=='novel',dynamic_loss_masked=True,
                        ego_to_world=np.asarray(ego).tolist(),camera_to_world=camera_pose.tolist(),K=K.tolist(),
                        before=legacy.save_image(stage/f'{key}_before.png',rgb),reference=legacy.save_image(stage/f'{key}_ref.png',ref),
                        raw_reference=asset(raw_path),mask=legacy.save_image(stage/f'{key}_mask.png',mask.astype(np.uint8)*255),
                        reference_mask=legacy.save_image(stage/f'{key}_reference_mask.png',ref_mask.astype(np.uint8)*255),
                        appearance_conditioning=data['_appearance_conditioning'],valid_fraction=float(mask.mean()),
                        target_actor_translation=(program['car_trans'][frame] if kind=='novel' else center).tolist())
                    prior=existing[(frame,cam,'swapped' if kind=='novel' else 'original')];verify_asset(prior['image'])
                    previous=legacy.image(prior['image']['path']);difference=float(np.abs(rgb.astype(float)-previous).mean()/255.)
                    row['source_preview_rgb_mean_difference']=difference
                    if difference>.005:raise ValueError('Render differs from accepted swap preview: '+key)
                    rows.append(row);write(stage/'rows'/f'{key}.json',row)
                print('Edited supervision frame',frame,'camera',cam,flush=True)
        legacy.validate_rows(rows)
        write(stage/'manifest.json',dict(completed=True,rows=rows,revision='mixed explicit original=0/swapped=1',weather='original',
            job=asset(out/'job.json'),trajectory=asset(out/'compiled_trajectory.json'),gpu_executed=True,
            edit_program=contract['program'],variant_contract='original replay + edited new views, explicit per-row revision'))
        module.enrich_render_manifest(out,job)
    finally:renderer.close()
    return dict(rendered_rows=len(rows),manifest=str(stage/'manifest.json'))

def completed(stage,out):
    path=out/{'render':'render/manifest.json','pseudo-gt':'pseudo_gt/manifest.json','train':'structure/train/receipt.json',
        'evaluate':'structure/evaluation/report.json','present':'swap_presentation/report.json'}[stage]
    if not path.exists():return False
    r=read(path)
    if r.get('completed') is not True:raise ValueError('Incomplete stage retained: '+str(path))
    if 'checkpoint' in r:verify_asset(r['checkpoint'])
    return True

def dispatch(stage,output):
    out=Path(output).resolve()
    if stage=='render':
        if completed(stage,out):configure(out);return dict(reused=True,stage=stage)
        return render(out)
    out,job,plan,module,contract,program,Renderer=configure(out)
    if completed(stage,out):return dict(reused=True,stage=stage)
    if stage=='pseudo-gt':
        result=module.legacy.pseudo_gt(out);module.conditioned_rows(out,job);return result
    if stage=='train':
        result=module.quality.train(out,'structure');receipt=read(out/'structure/train/receipt.json')
        write(out/'swap_candidate.json',dict(schema='position_swap_distilled_candidate_v1',checkpoint=receipt['checkpoint'],
            scene_manifest=receipt['scene_manifest'],program=contract['program'],conditioning=job['conditioning'],
            renderer_adapter=str(HERE/'workflow.py'),requires_swap_renderer=True,generic_scene_manifest_is_not_sufficient=True,
            automatic_promotion=False,default_model_changed=False))
        return result
    if stage=='evaluate':
        result=module.quality.evaluate(out,'structure');path=out/'structure/evaluation/report.json';report=read(path)
        report.update(position_swap_state_used=True,edit_program=contract['program'],fresh_edit_aware_pseudo_gt=True,default_model_changed=False)
        write(path,report);return result
    if stage=='present':
        from presentation import build
        return build(out)
    raise ValueError(stage)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','status','recover-render','recover-smpl','render','pseudo-gt','train','evaluate','present']);p.add_argument('--output',default=str(OUTPUT));a=p.parse_args()
    if a.stage=='recover-smpl':
        from recover_smpl import recover
        result=recover(a.output)
    elif a.stage=='recover-render':
        from recover_render import recover
        result=recover(a.output)
    elif a.stage=='prepare':result=prepare(a.output)
    elif a.stage=='status':
        configure(a.output);result=dict(output=a.output,stages={s:completed(s,Path(a.output)) for s in ['render','pseudo-gt','train','evaluate','present']})
    else:result=dispatch(a.stage,a.output)
    print(__import__('json').dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
