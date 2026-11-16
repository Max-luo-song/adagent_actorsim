"""Failure and routing tests for frozen material compatibility, no GPU required."""
import ast
import logging
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from pbr_validation import pbr, Config, REPO, vanilla_class, cpu_sh, compile_nodes, archived_formula
from test_waymo_adapter import extract


def state(count=4):
    generator=torch.Generator().manual_seed(42)
    shapes=dict(_means=(3,),_scales=(3,),_quats=(4,),_features_dc=(3,),_features_rest=(15,3),_opacities=(1,)) | pbr.MATERIAL_SHAPES
    return {key:torch.randn((count,)+shape,generator=generator) for key,shape in shapes.items()}


def model():
    return vanilla_class(cpu_sh)(class_name='Background',ctrl=Config(use_pbr=True,sh_degree=3,sh_degree_interval=1000),device=torch.device('cpu'))


class FrozenPBRTests(unittest.TestCase):
    def test_strict_load_preserves_all_material_and_rejects_unknown(self):
        value=state(); m=model()
        result=m.load_state_dict(value,strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        self.assertEqual(set(m.state_dict()),set(value))
        for name,tensor in m.state_dict().items():
            self.assertTrue(torch.equal(tensor,value[name]))
        with self.assertRaises(RuntimeError):
            model().load_state_dict(value | {'unknown_pretrained_parameter':torch.zeros(1)},strict=True)

    def test_no_missing_or_malformed_material_replacement(self):
        for name in pbr.MATERIAL_SHAPES:
            value=state();del value[name]
            with self.assertRaisesRegex(ValueError,name):model().load_state_dict(value,strict=True)
        value=state();value['_asg_dir']=torch.zeros(4,3,3)
        with self.assertRaisesRegex(ValueError,'_asg_dir'):model().load_state_dict(value,strict=True)

    def test_actual_gaussian_path_and_prune_keep_material_aligned(self):
        m=model(); m.load_state_dict(state(),strict=True); m.eval()
        cam=SimpleNamespace(camtoworlds=torch.eye(4))
        light=torch.ones(9,3)
        actual=m.get_gaussians(cam,global_sh_coeffs=light)
        expected=archived_formula(cpu_sh)(m,cam,light)
        torch.testing.assert_close(actual['_rgbs'],expected,rtol=0,atol=0)
        before={key:value.clone() for key,value in m.state_dict().items()}
        mask=torch.tensor([False,True,False,True]);m.prune_points(mask)
        for key,value in m.state_dict().items():
            torch.testing.assert_close(value,before[key][~mask],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'weather particles'):
            m.get_gaussians_with_particles(cam,torch.eye(4))
        with self.assertRaisesRegex(ValueError,'SH-only PLY'):
            m.add_instance_with_ply('not_read.ply',[0,0,0],torch.zeros(1,3),[0,1,1],[0,0,0])

    def test_lighting_preserves_historical_time_denominator(self):
        class Capture(nn.Module):
            def __init__(self):super().__init__();self.appearance_embedding=nn.Embedding(594,16)
            def forward(self,time,index):self.time=time;self.index=index;return torch.zeros(1,9,3)
        capture=Capture()
        pbr.lighting_for_frame(capture,197,593,594,'cpu')
        self.assertAlmostEqual(capture.time.item(),197/593,places=7)
        self.assertEqual(capture.index.item(),593)
        for idx in (-1,594):
            with self.assertRaises(ValueError):pbr.lighting_for_frame(capture,0,idx,594,'cpu')

    def test_collect_passes_light_to_pbr_without_enabling_weather(self):
        calls=[]
        class Gaussian:
            def __init__(self,pbr=False):self.use_pbr=pbr
            def get_gaussians(self,cam,**kwargs):
                calls.append(kwargs)
                return dict(_means=torch.zeros(1,3),_scales=torch.ones(1,3),_quats=torch.ones(1,4),
                            _rgbs=torch.ones(1,3),_opacities=torch.ones(1,1))
        coefficients=torch.ones(9,3)
        light=SimpleNamespace()
        fn=extract('models/trainers/base.py','BasicTrainer','collect_gaussians',
            dict(torch=torch,dataclass_gs=lambda **kw:SimpleNamespace(**kw),
                 lighting_for_frame=lambda *a:coefficients))
        source=SimpleNamespace(models={'GlobalLighting':light,'Background':Gaussian(True),'RigidNodes':Gaussian()},
            cur_frame=torch.tensor(1),num_full_images=6,device='cpu',particle_cfg=None,
            gaussian_classes={'Background':0,'RigidNodes':1},render_dynamic_mask=False)
        result,mask=fn(source,SimpleNamespace(),torch.tensor(5),edit_cfg={'Nodes':{}})
        self.assertIs(calls[0]['global_sh_coeffs'],coefficients)
        self.assertEqual(calls[1],{})
        self.assertEqual(result._means.shape,(2,3))
        self.assertEqual(mask.tolist(),[False,False])

    def test_strict_trainer_rejects_missing_and_extra_nodes(self):
        path=REPO/'models/trainers/base.py'
        tree=ast.parse(path.read_text())
        parent=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BasicTrainer')
        method=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='load_state_dict')
        cls=ast.ClassDef(name='BasicTrainer',bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
        actual=compile_nodes(path,[cls],dict(nn=nn,logger=logging.getLogger(__name__)))['BasicTrainer']
        for keys in ({}, {'Background':{},'ExtraNode':{}}):
            target=actual();target.models={'Background':nn.Module()}
            with self.assertRaisesRegex(ValueError,'node mismatch'):
                target.load_state_dict({'step':0,'models':keys},strict=True)

if __name__=='__main__':unittest.main()
