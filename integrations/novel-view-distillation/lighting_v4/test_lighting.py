
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
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from policy import build_tables,fallback_coefficients,override_lighting
torch.set_num_threads(1)


class ToyLight(torch.nn.Module):
    def __init__(self,n=12):
        super().__init__();self.appearance_embedding=torch.nn.Embedding(n,1)
        with torch.no_grad():self.appearance_embedding.weight[:,0]=torch.arange(n)
    def forward(self,time,ids):return (self.appearance_embedding(ids)+time).reshape(-1,1,1)


class Tests(unittest.TestCase):
    def setup_tables(self,frames=(0,2,3)):
        model=ToyLight();cameras={c:SimpleNamespace(unique_img_idx=torch.arange(4)*3+c) for c in [0,1,2]}
        return model,build_tables(model,cameras,frames,11)

    def test_interpolation_uses_same_camera_trained_codes_and_own_times(self):
        model,tables=self.setup_tables();value,audit=fallback_coefficients(tables,'interpolate',1,1)
        expected=(1.+(7.+2/11))/2
        self.assertAlmostEqual(float(value),expected,places=5)
        self.assertEqual(audit['source_frames'],[0,2]);self.assertEqual(audit['source_image_ids'],[1,7])
        self.assertEqual(audit['weights'],[.5,.5])

    def test_camera_mean_excludes_heldout_and_other_cameras(self):
        _,tables=self.setup_tables()
        value,audit=fallback_coefficients(tables,'camera_mean',1,2)
        self.assertAlmostEqual(float(value),np.mean([2.,8.+2/11,11.+3/11]),places=5)
        self.assertEqual(audit['source_image_ids'],[2,8,11]);self.assertNotIn(5,audit['source_image_ids'])

    def test_training_frames_and_direct_mode_unchanged(self):
        _,tables=self.setup_tables()
        for mode in ['direct','interpolate','camera_mean']:
            value,audit=fallback_coefficients(tables,mode,2,0)
            self.assertIsNone(value);self.assertFalse(audit['used_fallback'])
        self.assertIsNone(fallback_coefficients(tables,'direct',1,0)[0])

    def test_boundary_falls_back_to_nearest_trained_frame(self):
        _,tables=self.setup_tables(frames=(1,2))
        for query,source in [(0,1),(3,2)]:
            _,audit=fallback_coefficients(tables,'interpolate',query,0)
            self.assertEqual(audit['source_frames'],[source,source])

    def test_hook_restored_after_exception_and_parameters_unchanged(self):
        model,tables=self.setup_tables();before={k:v.clone() for k,v in model.state_dict().items()}
        args=(torch.tensor([[1/11]]),torch.tensor([4]));original=model(*args).detach()
        value,_=fallback_coefficients(tables,'interpolate',1,1)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with override_lighting(model,value):
                self.assertTrue(torch.equal(model(*args),value));raise RuntimeError('test')
        self.assertTrue(torch.equal(original,model(*args)))
        for key,t in model.state_dict().items():self.assertTrue(torch.equal(t,before[key]))

    def test_invalid_tables_or_mode_rejected(self):
        model,_=self.setup_tables()
        with self.assertRaises(ValueError):build_tables(model,{},[],11)
        with self.assertRaises(ValueError):build_tables(model,{0:SimpleNamespace(unique_img_idx=[0,999])},[1],11)
        _,tables=self.setup_tables()
        with self.assertRaises(ValueError):fallback_coefficients(tables,'invalid',1,0)

    def test_full_comparison_keeps_training_frames_and_reuses_only_identical_results(self):
        import evaluate as engine
        from contracts import write,read
        import pipeline as v1
        class FakeRenderer:
            def __init__(self,*args):
                light=ToyLight(597)
                with torch.no_grad():
                    light.appearance_embedding.weight.fill_(.5)
                    for f in range(10,199,10):light.appearance_embedding.weight[f*3:f*3+3]=.1
                light.forward=lambda time,ids:light.appearance_embedding(ids).reshape(-1,1,1)
                self.trainer=SimpleNamespace(models={'GlobalLighting':light},num_full_images=597)
                cameras={c:SimpleNamespace(unique_img_idx=torch.arange(199)*3+c) for c in [0,1,2]}
                self.backend=SimpleNamespace(cameras_data=cameras,dataset=SimpleNamespace(train_timesteps=[f for f in range(199) if f==0 or f%10],test_timesteps=list(range(10,199,10))))
            def frame(self,pose,frame,cam):
                return dict(cam_infos={'camera_to_world':torch.eye(4)},image_infos=dict(normed_time=torch.tensor([frame/198]),img_idx=torch.full((8,8),frame,dtype=torch.long)))
            def rgb(self,data):
                value=self.trainer.models['GlobalLighting'](data['image_infos']['normed_time'][:,None],data['image_infos']['img_idx'].flatten()[:1])
                return {'rgb':value.expand(8,8,3)}
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/'source';source.mkdir();write(source/'plan.json',{});write(base/'old.json',{})
            write(base/'job.json',{});(base/'checkpoint.bin').write_bytes(b'fixture')
            Image.new('RGB',(8,8),(128,)*3).save(base/'gt.png');Image.new('L',(8,8),255).save(base/'mask.png')
            samples=[];cached={}
            for f in engine.FRAMES:
                for c in [0,1,2]:
                    samples.append(dict(source_frame=f,camera_id=c,ego_to_world=np.eye(4).tolist(),target=v1.asset(base/'gt.png'),valid_mask=v1.asset(base/'mask.png')))
                    held=f in [30,60];delta=abs((.1 if held else .5)-128/255)
                    cached[f,c]=dict(source_dataset_test_frame=held,old={'l1':.15,'psnr':16.5},training_id_result=dict(l1=delta,psnr=-20*np.log10(delta)))
            plan=dict(assets={'source_checkpoint':v1.asset(base/'checkpoint.bin'),'job.json':v1.asset(base/'job.json')},parent_job=dict(inputs={'scene_manifest':{'path':'fixture'}},task={'renderer_bundle':'fixture'}))
            with patch.object(engine,'source_data',return_value=(plan,samples,cached)),patch.object(engine,'Renderer',FakeRenderer):
                result=engine.run(base/'output',source=source,id_ab=base/'old.json')
            self.assertEqual(result['actual_gpu_renders'],39)
            self.assertTrue(result['direct_baseline_reproduced'])
            self.assertEqual(result['recommended_for_data_regeneration'],'interpolate')
            self.assertEqual(result['summary']['source_train']['interpolate'],result['summary']['source_train']['direct'])
            report=read(base/'output/report.json')
            self.assertEqual(sum(r['render_executed']['interpolate'] for r in report['rows']),6)
            self.assertFalse(report['weights_updated'])
            self.assertEqual(read(base/'output/conditioning_policy.json')['source_holdout_policy'],'interpolate')


if __name__=='__main__':unittest.main()
