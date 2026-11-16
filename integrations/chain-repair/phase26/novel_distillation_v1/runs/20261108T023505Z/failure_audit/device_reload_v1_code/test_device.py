
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
from framework import align_trainer_device


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


if __name__=='__main__':unittest.main()
