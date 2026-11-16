
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
from types import SimpleNamespace
from unittest.mock import patch
import copy
import tempfile
import unittest
import numpy as np
import torch
import workflow as w
torch.set_num_threads(1)


class Light(torch.nn.Module):
    def __init__(self):
        super().__init__();self.appearance_embedding=torch.nn.Embedding(9,1)
        with torch.no_grad():self.appearance_embedding.weight[:,0]=torch.arange(9)/10
    def forward(self,time,ids):return (self.appearance_embedding(ids)+time).reshape(-1,1,1)


class Tests(unittest.TestCase):
    def context(self,light):
        return dict(global_lighting_fingerprint=w.lighting_fingerprint(light),policy_asset={'sha256':'verified'},
            policy={'source_train_frames':[0,2],'time_denominator':8,'source_holdout_policy':'interpolate'})

    def test_lighting_fingerprint_detects_real_weight_change(self):
        model=Light();a=w.lighting_fingerprint(model)
        with torch.no_grad():model.appearance_embedding.weight[0]+=1
        self.assertNotEqual(a,w.lighting_fingerprint(model))

    def make_renderer(self,light):
        def initialize(instance,*args,**kwargs):
            cameras={c:SimpleNamespace(unique_img_idx=torch.arange(3)*3+c) for c in [0,1,2]}
            instance.trainer=SimpleNamespace(models={'GlobalLighting':light},num_full_images=9)
            instance.backend=SimpleNamespace(dataset=SimpleNamespace(train_timesteps=[0,2]),cameras_data=cameras)
        return patch.object(w.BaseRenderer,'__init__',initialize)

    def test_renderer_rejects_wrong_lighting_or_partition(self):
        model=Light();ctx=self.context(model)
        with self.make_renderer(model),patch.object(w,'CONTEXT',ctx):
            with torch.no_grad():model.appearance_embedding.weight[0]+=1
            with self.assertRaisesRegex(ValueError,'GlobalLighting'):w.ConditionedRenderer('scene','bundle')
        model=Light();ctx=self.context(model);ctx['policy']['source_train_frames']=[0,1,2]
        with self.make_renderer(model),patch.object(w,'CONTEXT',ctx):
            with self.assertRaisesRegex(ValueError,'split'):w.ConditionedRenderer('scene','bundle')

    def test_same_conditioning_in_render_and_gradient_path(self):
        model=Light();model.requires_grad_(False);ctx=self.context(model);albedo=torch.nn.Parameter(torch.tensor(.5))
        def frame(instance,pose,frame,cam):
            return dict(cam_infos={'camera_to_world':torch.eye(4)},image_infos=dict(img_idx=torch.full((8,8),frame,dtype=torch.long),normed_time=torch.tensor([frame/2])))
        def rgb(instance,data,layers=False):
            ids=data['image_infos']['img_idx'].flatten()[:1]
            light=model(torch.tensor([[1/8]]),ids)
            return {'rgb':albedo*light.expand(8,8,3)}
        w.RENDER_AUDIT.clear();w.SOURCE_MAPS.clear()
        with self.make_renderer(model),patch.object(w,'CONTEXT',ctx),patch.object(w.BaseRenderer,'frame',frame),patch.object(w.BaseRenderer,'rgb',rgb):
            renderer=w.ConditionedRenderer('scene','bundle');data=renderer.frame(np.eye(4),1,1)
            self.assertTrue((data['image_infos']['img_idx']==4).all())
            out=renderer.rgb(data)['rgb'];out.mean().backward()
            self.assertGreater(float(albedo.grad),0)
            self.assertEqual(data['_appearance_conditioning']['lighting']['source_frames'],[0,2])
            self.assertEqual(w.RENDER_AUDIT[-1]['policy_sha256'],'verified')
            self.assertFalse(bool(model._forward_hooks))

    def test_actual_render_receipts_enriched_for_both_paths(self):
        from contracts import write,read
        model=Light();ctx=self.context(model);audit=dict(source_frame=1,camera_id=1,image_id=4,policy_sha256='verified')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'render/rows').mkdir(parents=True)
            rows=[dict(id=k,source_frame=1,camera_id=1) for k in ['new','original']]
            write(out/'render/manifest.json',dict(completed=True,rows=rows))
            maps={'camera_image_ids':{'1':[1,4,7]}}
            with patch.object(w,'RENDER_AUDIT',[audit,copy.deepcopy(audit)]),patch.object(w,'SOURCE_MAPS',[maps]):
                w.enrich_render_manifest(out,dict(conditioning=ctx))
            saved=read(out/'render/manifest.json')
            self.assertEqual(saved['rows'][0]['appearance_conditioning'],saved['rows'][1]['appearance_conditioning'])
            self.assertTrue((out/'render/rows/new.json').is_file())

    def test_old_or_mixed_supervision_is_rejected(self):
        from contracts import write
        model=Light();ctx=self.context(model)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'render').mkdir()
            source=out/'render/conditioning_sources.json';write(source,{'camera_image_ids':{'1':[1,4,7]}})
            valid=dict(schema='render_appearance_conditioning_v1',image_id=4,policy_sha256='verified',global_lighting_fingerprint=ctx['global_lighting_fingerprint'])
            write(out/'render/manifest.json',dict(appearance_conditioning_sources=w.legacy.asset(source),conditioning_policy=ctx['policy_asset'],rows=[dict(id='row',appearance_conditioning=valid)]))
            row=dict(id='row',source_frame=1,camera_id=1)
            with patch.object(w.legacy,'target_rows',return_value=[row]):
                with self.assertRaisesRegex(ValueError,'Mixed/old'):w.conditioned_rows(out,dict(conditioning=ctx))
            row['appearance_conditioning']=valid
            with patch.object(w.legacy,'target_rows',return_value=[row]):
                self.assertEqual(len(w.conditioned_rows(out,dict(conditioning=ctx))),1)

    def test_fresh_data_all_stages_and_candidate_reload_use_one_conditioning(self):
        from contracts import read,write
        from PIL import Image
        class Color(torch.nn.Module):
            def __init__(self,pbr=False):
                super().__init__();self.use_pbr=pbr
                for k in ['_features_dc','_features_rest','_means']:
                    setattr(self,k,torch.nn.Parameter(torch.zeros(1,3)))
        class FakeRenderer:
            def __init__(self,scene,bundle):
                self.trainer=torch.nn.Module();self.trainer.models={'Background':Color(True),'RoadNodes':Color(),'GlobalLighting':Light()}
                checkpoint=read(read(scene)['source_report'])['checkpoint']
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                for name,model in self.trainer.models.items():model.load_state_dict(state['models'][name])
            def frame(self,*args):return {}
            def rgb(self,data):
                b=self.trainer.models['Background'];r=self.trainer.models['RoadNodes']
                return {'rgb':(.5*b._features_dc+.5*r._features_dc+.1*r._features_rest).sigmoid().expand(16,16,3)}
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);out=base/'job';models={'Background':Color(True),'RoadNodes':Color(),'GlobalLighting':Light()}
            torch.save(dict(models={k:m.state_dict() for k,m in models.items()},step=40000),base/'checkpoint.pth')
            write(base/'config.yaml',{});write(base/'source.json',dict(checkpoint=str(base/'checkpoint.pth')))
            write(base/'scene.json',dict(source_report=str(base/'source.json'),vehicle_config=str(base/'vehicle.json')))
            write(base/'task.json',{})
            write(base/'acceptance.json',dict(recommended_for_data_regeneration='interpolate',quality_checks={'interpolate':{'passed':True}}))
            write(base/'policy.json',dict(schema='scene_appearance_conditioning_v1',source_training_frame_policy='direct',source_holdout_policy='interpolate',
                checkpoint=w.legacy.asset(base/'checkpoint.pth'),validation_report=w.legacy.asset(base/'acceptance.json'),
                source_train_frames=[0,2],time_denominator=8))
            for name,value in [('before',128),('target',153),('mask',255)]:Image.new('RGB',(16,16),(value,)*3).save(base/f'{name}.png')
            def prepare(task,output):
                output=Path(output);output.mkdir()
                inputs={k:w.legacy.asset(base/name) for k,name in [('checkpoint','checkpoint.pth'),('config','config.yaml'),('source_report','source.json'),('scene_manifest','scene.json'),('task','task.json')]}
                write(output/'compiled_trajectory.json',{'fps':10})
                write(output/'job.json',dict(inputs=inputs,implementation=w.family_pins(),trajectory_sha256=w.sha(output/'compiled_trajectory.json'),task={'renderer_bundle':'fixture'}))
                return {'prepared':True,'samples_by_split':{'train':3,'eval':3}}
            def render(output):
                output=Path(output);(output/'render/rows').mkdir(parents=True);job=read(output/'job.json');rows=[]
                ctx=job['conditioning']
                for f,split in [(0,'train'),(1,'eval')]:
                    for cam in [0,1,2]:
                        for kind in ['novel','original']:
                            audit=dict(source_frame=f,camera_id=cam,image_id=3*f+cam,policy_sha256=ctx['policy_asset']['sha256'],global_lighting_fingerprint=ctx['global_lighting_fingerprint'])
                            w.RENDER_AUDIT.append(audit)
                            rows.append(dict(id=f'{f}_{cam}_{kind}',source_frame=f,camera_id=cam,kind=kind,split=split,scene=551,edit_revision=0,
                                checkpoint_sha256=job['inputs']['checkpoint']['sha256'],ego_to_world=np.eye(4).tolist(),camera_to_world=np.eye(4).tolist(),K=np.eye(3).tolist(),
                                before=w.legacy.asset(base/'before.png'),reference=w.legacy.asset(base/'target.png'),mask=w.legacy.asset(base/'mask.png')))
                w.SOURCE_MAPS.append(dict(camera_image_ids={str(c):[c,c+3,c+6] for c in [0,1,2]}))
                write(output/'render/manifest.json',dict(completed=True,rows=rows,job=w.legacy.asset(output/'job.json'),trajectory=w.legacy.asset(output/'compiled_trajectory.json')))
                return {'rendered':True}
            def teacher(output):
                output=Path(output);(output/'pseudo_gt').mkdir();rows=read(output/'render/manifest.json')['rows']
                for r in rows:r.update(accepted=True,target=w.legacy.asset(base/'target.png'),valid_mask=w.legacy.asset(base/'mask.png'),target_kind='real_original_gt' if r['kind']=='original' else 'difix_pseudo_gt')
                write(output/'pseudo_gt/manifest.json',dict(completed=True,identity_fallback=False,render_manifest=w.legacy.asset(output/'render/manifest.json'),rows=rows))
                return {'pseudo_gt':True}
            with patch.object(w.legacy,'prepare',side_effect=prepare),patch.object(w.legacy,'render_pack',side_effect=render),patch.object(w.legacy,'pseudo_gt',side_effect=teacher),\
                 patch.object(w,'ConditionedRenderer',FakeRenderer),patch.object(w,'CONTEXT',None),\
                 patch.object(w.framework,'Renderer',w.framework.Renderer),patch.object(w.quality,'Renderer',w.quality.Renderer),\
                 patch.object(w.quality,'load_plan',w.quality.load_plan),patch.object(w.legacy,'implementation',w.legacy.implementation),\
                 patch.object(w.legacy,'make_comparison',side_effect=lambda panels,stage,fps:(stage/'before_teacher_after.mp4').write_bytes(b'fixture')):
                w.prepare(base/'task.json',out,base/'policy.json')
                self.assertEqual(read(out/'plan.json')['parent'],str(out))
                w.dispatch('render',out);w.dispatch('pseudo-gt',out)
                self.assertEqual(len(w.conditioned_rows(out,read(out/'job.json'))),12)
                result=w.dispatch('train',out);self.assertEqual(result['steps'],72)
                w.dispatch('evaluate',out)
                report=read(out/'structure/evaluation/report.json')
                self.assertTrue(report['fresh_pseudo_gt']);self.assertFalse(report['geometry_changed'])
                self.assertFalse(report['old_v2_absolute_pseudo_gt_metrics_comparable'])
                receipt=read(out/'structure/train/receipt.json')
                self.assertTrue(all('_novel' in i or '_original' in i for i in receipt['trained_sample_ids']))
                self.assertEqual(receipt['active_gradient_steps']['Background._features_dc'],72)


if __name__=='__main__':unittest.main()
