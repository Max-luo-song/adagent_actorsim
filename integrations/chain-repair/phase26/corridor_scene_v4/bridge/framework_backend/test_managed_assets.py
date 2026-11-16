
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
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from managed_assets import ManagedAssets, load_asset
from backend import FrameworkSceneBackend
from managed_evaluation import ManagedActorEvaluation


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        means = torch.tensor([[1.,0,2],[2.,-.5,3],[0.,-.2,1]])
        params = (3, means, torch.zeros(3,1,3), torch.full((3,15,3), .01), torch.empty(3,0),
                  torch.zeros(3,3), torch.tensor([[1.,0,0,0]]).repeat(3,1), torch.zeros(3,1), 1.)
        torch.save((params,0), self.path/'gs.pth')
        self.edit = SimpleNamespace(operation='insert', object_id='test', asset=self.path,
            calibration={'support':{'raw_recenter_xz':[1.,2.],'wlh':[2.,4.,1.5],'bottom_y':0.}},
            asset_to_world=np.eye(4))
        self.base = SimpleNamespace(_means=torch.zeros(2,3), _rgbs=torch.zeros(2,3),
            _quats=torch.tensor([[1.,0,0,0]]).repeat(2,1), _scales=torch.ones(2,3), _opacities=torch.ones(2,1))
        self.trainer = SimpleNamespace(gaussian_classes={'Background':0,'RigidNodes':1})
        def original(*args, **kwargs):
            self.trainer.pts_labels = torch.tensor([0,1])
            return self.base, torch.zeros(2,dtype=torch.bool)
        self.trainer.collect_gaussians = original
        self.registry = ManagedAssets(self.trainer, 'cpu')

    def test_load_recenter_activations_and_rotation_covariance(self):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_euler('z',90,degrees=True).as_matrix()
        pose[:3,3] = [10,20,1]
        self.edit.asset_to_world = pose
        asset = load_asset(self.edit,'cpu')
        f = asset.fields(torch.tensor([0.,0.,0.]))
        np.testing.assert_allclose(f['_means'][0].numpy(), pose[:3,3],atol=1e-6)
        self.assertTrue(torch.all(f['_scales']==1))
        self.assertTrue(torch.all(f['_opacities']==.5))
        q = f['_quats'][0].numpy()
        np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),pose[:3,:3],atol=1e-6)
        self.assertTrue(all(torch.isfinite(v).all() for v in f.values()))

    def test_local_sh_appearance_rotates_with_asset(self):
        plain = load_asset(self.edit,'cpu')
        camera = torch.tensor([-3.,1.,4.])
        before = plain.fields(camera)['_rgbs']
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_euler('xyz',[10,20,80],degrees=True).as_matrix()
        pose[:3,3] = [8,2,5]
        self.edit.asset_to_world = pose
        rotated = load_asset(self.edit,'cpu')
        new_camera = torch.tensor(camera.numpy() @ pose[:3,:3].T + pose[:3,3],dtype=torch.float32)
        torch.testing.assert_close(before,rotated.fields(new_camera)['_rgbs'],atol=1e-6,rtol=0)

    def test_merge_labels_remove_and_no_accumulated_pose_transform(self):
        cam = SimpleNamespace(camtoworlds=torch.eye(4))
        base, _ = self.registry.collect(cam=cam)
        self.assertIs(base,self.base)
        self.registry.apply(self.edit)
        first = self.registry.assets['test']
        pose = np.eye(4);pose[0,3]=5
        moved = SimpleNamespace(operation='set_pose',object_id='test',asset_to_world=pose)
        self.registry.apply(moved);self.registry.apply(moved)
        torch.testing.assert_close(self.registry.assets['test'].local_means,first.local_means)
        merged,freeze = self.registry.collect(cam=cam)
        self.assertEqual(len(merged._means),5)
        self.assertEqual(self.trainer.pts_labels.tolist(),[0,1,2,2,2])
        self.assertEqual(len(freeze),5)
        self.assertEqual(len(self.base._means),2)
        self.registry.apply(SimpleNamespace(operation='remove',object_id='test'))
        restored,_ = self.registry.collect(cam=cam)
        self.assertIs(restored,self.base)
        self.assertEqual(self.trainer.pts_labels.tolist(),[0,1])
        original_collect = self.registry.original_collect
        self.registry.close()
        self.assertEqual(self.trainer.collect_gaussians, original_collect)
        self.assertIsNone(self.registry.trainer)

    def test_history_clear_preserves_asset_full_reset_removes_it(self):
        backend = FrameworkSceneBackend.__new__(FrameworkSceneBackend)
        backend.torch=torch;backend._last=None;backend.managed_assets=self.registry
        backend.apply(self.edit)
        backend.clear_observation_history()
        self.assertIn('test',self.registry.assets)
        backend.reset()
        self.assertFalse(self.registry.assets)

    def test_failed_history_clear_rolls_back_managed_registry(self):
        backend = FrameworkSceneBackend.__new__(FrameworkSceneBackend)
        backend.torch=torch;backend._last=None;backend.managed_assets=self.registry
        def failed():raise RuntimeError('worker failed')
        backend.clear_observation_history=failed
        with self.assertRaises(RuntimeError):backend.apply(self.edit)
        self.assertFalse(self.registry.assets)

    def test_invalid_asset_does_not_replace_existing_registry(self):
        self.registry.apply(self.edit)
        before = dict(self.registry.assets)
        self.edit.object_id='bad'
        params,_=torch.load(self.path/'gs.pth',weights_only=False)
        params[1][0,0]=float('nan')
        torch.save((params,0),self.path/'gs.pth')
        with self.assertRaises(ValueError):self.registry.apply(self.edit)
        self.assertEqual(list(self.registry.assets),list(before))

    def test_proxy_corners_use_the_same_pose_as_render(self):
        self.edit.asset_to_world=np.array([[1,0,0,5],[0,0,1,3],[0,-1,0,0],[0,0,0,1]],float)
        asset=load_asset(self.edit,'cpu')
        corners=asset.corners()
        np.testing.assert_allclose(corners.mean(0),[5,3,.75])
        self.assertEqual(corners[:,2].min(),0.)
        self.assertEqual(corners[:,2].max(),1.5)

    def test_managed_collision_rows_and_vertical_filter(self):
        self.edit.asset_to_world=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
        self.registry.apply(self.edit)
        native=SimpleNamespace(groups={},check=lambda *args:dict(nearest=dict(gap_m=10.,overlap=False),overlaps=[],overlap=False))
        evaluator=ManagedActorEvaluation(native,self.registry)
        vehicle=SimpleNamespace(half_length_m=1.5,half_width_m=.8,body_height_m=1.7)
        module=SimpleNamespace(polygon_check=lambda a,b:dict(gap_m=-1.,overlap=True))
        with patch.dict(sys.modules,{'closed172_common':module}):
            result=evaluator.check(np.eye(4),80,vehicle)
            self.assertTrue(result['overlap'])
            self.assertEqual(result['nearest']['object_id'],'managed:test')
            pose=self.edit.asset_to_world.copy();pose[2,3]=100
            self.registry.apply(SimpleNamespace(operation='set_pose',object_id='test',asset_to_world=pose))
            result=evaluator.check(np.eye(4),80,vehicle)
            self.assertFalse(result['overlap'])
            self.assertFalse(result['managed_reference'])


if __name__=='__main__':unittest.main()
