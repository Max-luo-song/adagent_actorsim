
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

import copy
pass  # Path is provided by the portable boundary above
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from quality_policy import PROFILES,select_parameters,epoch_schedule,loss_terms,objective,clamp_color_update,parameter_stats


class Model(torch.nn.Module):
    def __init__(self,pbr=False):
        super().__init__();self.use_pbr=pbr
        for field in ['_features_dc','_features_rest','_means','_scales','_opacities']:
            setattr(self,field,torch.nn.Parameter(torch.zeros(1,3)))


class Tests(unittest.TestCase):
    def test_active_pbr_parameters_and_frozen_geometry(self):
        t=torch.nn.Module();t.models={'Background':Model(True),'RoadNodes':Model(),'RigidNodes':Model()}
        selected,groups=select_parameters(t,PROFILES['structure'])
        self.assertEqual(set(selected),{('Background','_features_dc'),('RoadNodes','_features_dc'),('RoadNodes','_features_rest')})
        self.assertFalse(t.models['Background']._features_rest.requires_grad)
        self.assertFalse(t.models['Background']._scales.requires_grad)
        self.assertFalse(t.models['RigidNodes']._features_dc.requires_grad)
        self.assertEqual([g['lr'] for g in groups],[.001,.001,.000125])

    def test_probe_matches_v1_sampling_and_budget(self):
        groups={k:[dict(kind=k,id=i) for i in range(51)] for k in ['novel','original']}
        rng=np.random.default_rng(20261108);actual=epoch_schedule(groups,PROFILES['lr_probe'],rng)
        rng=np.random.default_rng(20261108);expected=[groups[k][i] for k in ['novel','original','original'] for i in rng.permutation(51)]
        self.assertEqual(actual,expected);self.assertEqual(len(actual)*3,459)

    def test_balanced_and_structure_are_matched_except_loss(self):
        groups={k:[dict(kind=k,id=i) for i in range(51)] for k in ['novel','original']}
        a=epoch_schedule(groups,PROFILES['balanced'],np.random.default_rng(2))
        b=epoch_schedule(groups,PROFILES['structure'],np.random.default_rng(2))
        self.assertEqual(a,b);self.assertEqual(len(a)*12,1224)
        self.assertEqual([r['kind'] for r in a[:4]],['novel','original','novel','original'])
        for key in ['epochs','dc_lr','rest_lr','schedule']:self.assertEqual(PROFILES['balanced'][key],PROFILES['structure'][key])

    def test_identical_images_and_uniform_regions_have_finite_gradients(self):
        for value in [0.,.5,1.]:
            pred=torch.full((16,16,3),value,requires_grad=True);mask=torch.ones(16,16,dtype=torch.bool)
            terms=loss_terms(pred,pred.detach().clone(),mask);loss=objective(terms,PROFILES['structure'])
            self.assertLess(float(loss),1e-5);loss.backward();self.assertTrue(torch.isfinite(pred.grad).all())

    def test_outside_mask_cannot_affect_structural_supervision(self):
        target=torch.rand(32,32,3);pred=target.clone();mask=torch.zeros(32,32,dtype=torch.bool);mask[6:26,6:26]=True
        pred[~mask]=1-target[~mask];pred.requires_grad_()
        terms=loss_terms(pred,target,mask);loss=objective(terms,PROFILES['structure'])
        self.assertLess(float(loss),1e-5);self.assertEqual(terms['ssim_windows'],100)
        loss.backward();self.assertEqual(float(pred.grad[~mask].abs().sum()),0)

    def test_blur_has_structural_and_edge_error(self):
        y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        target=(((x//4+y//4)%2).float()*.8+.1)[...,None].expand(-1,-1,3)
        blur=F.avg_pool2d(target.permute(2,0,1)[None],5,1,2)[0].permute(1,2,0).clone().requires_grad_()
        terms=loss_terms(blur,target,torch.ones(32,32,dtype=torch.bool))
        self.assertGreater(float(terms['edge']),.05);self.assertGreater(float(terms['ssim']),.05)
        objective(terms,PROFILES['structure']).backward();self.assertTrue(torch.isfinite(blur.grad).all())

    def test_trust_region_clamps_color_updates_only(self):
        t=torch.nn.Module();t.models={'Background':Model(True),'RoadNodes':Model()}
        selected,groups=select_parameters(t,PROFILES['structure']);initial={k:p.detach().clone() for k,p in selected.items()}
        with torch.no_grad():
            for p in selected.values():p.add_(10)
        clamp_color_update(selected,initial)
        for k,p in selected.items():self.assertLessEqual(float(p.abs().max()),.250001 if k[1]=='_features_dc' else .100001)
        self.assertEqual(float(t.models['Background']._means.abs().sum()),0)

    def test_color_optimization_updates_and_logs_actual_adam_state(self):
        t=torch.nn.Module();t.models={'Background':Model(True)}
        selected,groups=select_parameters(t,PROFILES['structure']);p=selected['Background','_features_dc']
        initial={k:p.detach().clone() for k,p in selected.items()};opt=torch.optim.Adam(groups,eps=1e-8)
        mask=torch.ones(16,16,dtype=torch.bool);target=torch.full((16,16,3),.6);first=None
        for i in range(20):
            pred=p.sigmoid().expand(16,16,3);terms=loss_terms(pred,target,mask);loss=objective(terms,PROFILES['structure'])
            if first is None:first=float(loss)
            opt.zero_grad();loss.backward();opt.step();clamp_color_update(selected,initial)
        self.assertLess(float(loss),first)
        stats=parameter_stats(selected,initial,opt)['Background._features_dc']
        self.assertTrue(stats['gradient_present']);self.assertGreater(stats['sampled_delta_mean'],0)
        self.assertGreater(stats['adam_rms_to_eps_factor'],0)

    def test_empty_mask_and_malformed_inputs_rejected(self):
        with self.assertRaises(ValueError):loss_terms(torch.zeros(16,16,3),torch.zeros(16,16,3),torch.zeros(16,16,dtype=torch.bool))
        with self.assertRaises(ValueError):loss_terms(torch.zeros(16,16,3),torch.zeros(16,16,3),torch.ones(16,16))

    def test_train_save_reload_evaluate_and_compare_without_validation_leak(self):
        import experiment as engine
        from contracts import write,read
        import pipeline as legacy
        from PIL import Image
        class FakeRenderer:
            def __init__(self,scene,bundle):
                self.trainer=torch.nn.Module();self.trainer.models={'Background':Model(True),'RoadNodes':Model()}
                src=read(Path(read(scene)['source_report']))
                state=torch.load(src['checkpoint'],map_location='cpu',weights_only=False)
                for k,m in self.trainer.models.items():m.load_state_dict(state['models'][k])
            def frame(self,*args):return {}
            def rgb(self,data):
                b=self.trainer.models['Background'];r=self.trainer.models['RoadNodes']
                return {'rgb':(.5*b._features_dc+.5*r._features_dc+.1*r._features_rest).sigmoid().expand(16,16,3)}
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);parent=out/'parent';parent.mkdir()
            models={'Background':Model(True),'RoadNodes':Model()}
            base=dict(models={n:m.state_dict() for n,m in models.items()},step=40000)
            torch.save(base,parent/'base.pth');write(parent/'config.yaml',{})
            write(parent/'source.json',dict(checkpoint=str(parent/'base.pth')))
            write(parent/'scene.json',dict(source_report=str(parent/'source.json'),vehicle_config=str(parent/'vehicle.json')))
            write(parent/'compiled_trajectory.json',{'fps':10})
            (parent/'evaluation').mkdir();write(parent/'evaluation/report.json',dict(averages={'novel':{'after':{'l1':.1}}}))
            write(out/'plan.json',{'fixture':True})
            for name,value in [('before',128),('target',153),('mask',255)]:Image.new('RGB',(16,16),(value,)*3).save(parent/f'{name}.png')
            rows=[dict(id=f'{kind}_{split}_{cam}',kind=kind,split=split,camera_id=cam,source_frame=frame,
                accepted=True,ego_to_world=np.eye(4).tolist(),target=legacy.asset(parent/'target.png'),
                valid_mask=legacy.asset(parent/'mask.png'),before=legacy.asset(parent/'before.png'))
                for kind in ['novel','original'] for split,frame in [('train',0),('eval',4)] for cam in [0,1,2]]
            policy=dict(PROFILES['structure'],epochs=1)
            inputs={k:legacy.asset(parent/f) for k,f in [('source_checkpoint','base.pth'),('source_config','config.yaml')]}
            job_inputs={k:legacy.asset(parent/f) for k,f in [('scene_manifest','scene.json'),('source_report','source.json')]}
            plan=dict(profiles={'structure':policy},parent=str(parent),assets=inputs,
                parent_job=dict(inputs=job_inputs,task={'renderer_bundle':'fixture'}))
            def video(panels,stage,fps):(stage/'before_teacher_after.mp4').write_bytes(b'fixture')
            with patch.object(engine,'load_plan',return_value=(out,plan,rows)),patch.object(engine,'Renderer',FakeRenderer),patch.object(legacy,'make_comparison',side_effect=video):
                result=engine.train(out,'structure');self.assertEqual(result['steps'],6)
                receipt=read(out/'structure/train/receipt.json')
                self.assertTrue(all('_train_' in k for k in receipt['trained_sample_ids']))
                engine.evaluate(out,'structure');report=read(out/'structure/evaluation/report.json')
                self.assertEqual(len(report['metrics']),6)
                self.assertFalse(report['geometry_changed']);self.assertFalse(report['automatic_promotion'])
                comparison=engine.compare(out)
                self.assertFalse(report['quality_screen_passed'])
                self.assertIsNone(comparison['recommended_for_further_review'])
            saved=torch.load(out/'structure/train/checkpoint_distilled.pth',weights_only=False)
            self.assertTrue(torch.equal(base['models']['Background']['_scales'],saved['models']['Background']['_scales']))
            self.assertEqual(saved['step'],40000)


if __name__=='__main__':unittest.main()
