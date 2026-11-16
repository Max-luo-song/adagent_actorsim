
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

import sys
pass  # Path is provided by the portable boundary above
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from framework import align_trainer_device,trainer_tensor_placements


class Tests(unittest.TestCase):
    def trainer(self):
        trainer=torch.nn.Module();pbr=torch.nn.Module()
        pbr.register_parameter('_normals',torch.nn.Parameter(torch.tensor([[0.,0.,1.]])))
        pbr.register_buffer('frozen_buffer',torch.ones(1))
        trainer.models={'Background':pbr}  # deliberately unregistered plain dict
        return trainer

    def test_plain_dict_parameters_and_buffers_are_moved(self):
        trainer=self.trainer();self.assertEqual(len(list(trainer.parameters())),0)
        result=align_trainer_device(trainer,torch.device('meta'))
        self.assertEqual(result['Background'],2)
        self.assertEqual(trainer.models['Background']._normals.device.type,'meta')
        self.assertEqual(trainer.models['Background'].frozen_buffer.device.type,'meta')

    def test_same_device_preserves_values(self):
        trainer=self.trainer();before=trainer.models['Background']._normals.detach().clone()
        align_trainer_device(trainer,'cpu')
        self.assertTrue(torch.equal(before,trainer.models['Background']._normals))

    def test_incomplete_move_is_rejected(self):
        trainer=self.trainer();trainer.models['Background'].to=lambda device:None
        with self.assertRaisesRegex(RuntimeError,'Background._normals'):
            align_trainer_device(trainer,'meta')

    def test_plain_rigid_state_and_nested_smpl_state_are_moved(self):
        trainer=self.trainer();node=trainer.models['Background']
        node.instances_fv=torch.tensor([[True,False]])
        node.instances_size=torch.tensor([[2.,4.,1.]])
        node.point_ids=torch.tensor([0,1])
        node.cached={'valid':[node.instances_fv], 'sizes':(node.instances_size,)}
        align_trainer_device(trainer,'meta')
        for field in ['instances_fv','instances_size','point_ids']:
            self.assertEqual(getattr(node,field).device.type,'meta')
        self.assertIs(node.cached['valid'][0],node.instances_fv)
        self.assertIs(node.cached['sizes'][0],node.instances_size)
        self.assertTrue(all(t.device.type=='meta' for t in trainer_tensor_placements(trainer).values()))

    def test_nested_submodule_plain_tensor_is_moved(self):
        trainer=self.trainer();node=trainer.models['Background'];node.template=torch.nn.Module()
        node.template.parents=torch.tensor([0,0,1])
        node.template.caches=[{'weights':torch.ones(2)}]
        align_trainer_device(trainer,'meta')
        self.assertEqual(node.template.parents.device.type,'meta')
        self.assertEqual(node.template.caches[0]['weights'].device.type,'meta')

    def test_cpu_move_preserves_plain_values_dtype_and_aliases(self):
        trainer=self.trainer();node=trainer.models['Background']
        node.instances_fv=torch.tensor([[True,False]]);node.point_ids=torch.tensor([0,1])
        node.cached={'fv':node.instances_fv};before=node.instances_fv.clone()
        align_trainer_device(trainer,'cpu')
        self.assertTrue(torch.equal(node.instances_fv,before))
        self.assertEqual(node.instances_fv.dtype,torch.bool)
        self.assertEqual(node.point_ids.dtype,torch.int64)
        self.assertIs(node.cached['fv'],node.instances_fv)


if __name__=='__main__':unittest.main()
