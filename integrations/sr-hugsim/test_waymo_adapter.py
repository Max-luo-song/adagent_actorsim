"""CPU contract tests. Extract actual methods to avoid importing CUDA-only SR dependencies."""

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

import ast
import logging
import os
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from waymo_adapter import SR_TO_HUGSIM, sr_ego_pose, frame_index

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT/'integrations/sr-origin-main'


def extract(relative, klass, method, namespace):
    tree = ast.parse((REPO/relative).read_text())
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == klass)
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == method)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(REPO/relative), 'exec'), namespace)
    return namespace[method]


class Contracts(unittest.TestCase):
    def test_pose_converts_both_bases(self):
        source = np.eye(4); source[:3, 3] = [7, 2, 1]
        native = SR_TO_HUGSIM @ source @ SR_TO_HUGSIM.T
        np.testing.assert_allclose(sr_ego_pose(native), source)
        source[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        np.testing.assert_allclose(sr_ego_pose(SR_TO_HUGSIM @ source @ SR_TO_HUGSIM.T), source)
        with self.assertRaises(ValueError): sr_ego_pose(np.zeros((4, 4)))

    def test_clock_no_clamping(self):
        self.assertEqual(frame_index(.3, 10, 198), 3)
        for t in (-1, float('nan'), 19.8):
            with self.assertRaises(ValueError): frame_index(t, 10, 198)

    def test_real_calibrations_172_and_552(self):
        import json
        for report in (ROOT/'outputs').glob('waymo*three_setup*/source_report.json'):
            source = _hugsim_expand(json.loads(report.read_text()))
            if source['scene'] not in (172, 552): continue
            for cam in source.get('trained_cameras', source.get('camera_ids')):
                fake = SimpleNamespace(data_path=source['inputs'], cam_id=cam,
                    original_size=(1280, 1920), load_size=(320, 480),
                    start_timestep=source['start_timestep'], end_timestep=source['start_timestep']+2)
                cv = SR_TO_HUGSIM.T
                fn = extract('datasets/waymo/waymo_sourceloader.py', 'WaymoCameraData', 'load_calibrations',
                             dict(np=np, torch=torch, os=os, OPENCV2DATASET=cv))
                fn(fake)
                e0 = np.loadtxt(Path(source['inputs'])/'ego_pose'/f'{fake.start_timestep:03d}.txt')
                e1 = np.loadtxt(Path(source['inputs'])/'ego_pose'/f'{fake.start_timestep+1:03d}.txt')
                expected = np.linalg.inv(e0) @ e1 @ fake.cam_to_ego
                np.testing.assert_allclose(fake.cam_to_worlds[1], expected, atol=2e-6)
                self.assertEqual(tuple(fake.intrinsics.shape), (2, 3, 3))

    def test_camera_subset_keeps_training_embedding_order(self):
        class Camera:
            def __init__(self, **kw): self.kw=kw
            def __len__(self): return 4
            def load_time(self, t): pass
            def set_unique_ids(self, idx, ids): self.unique_cam_idx=idx; self.unique_img_idx=ids
        fn=extract('datasets/waymo/waymo_sourceloader.py', 'WaymoPixelSource', 'load_specified_cameras', dict(np=np, torch=torch, WaymoCameraData=Camera))
        source=SimpleNamespace(camera_list=[0, 1, 2], dataset_name='waymo', data_path='unused',
            start_timestep=0, end_timestep=4, buffer_downscale=1, device='cpu', normalized_time=torch.arange(4))
        cameras=fn(source,[2,0],[4,4])
        self.assertEqual(cameras[2].unique_img_idx.tolist(), [2,5,8,11])
        self.assertEqual(cameras[0].unique_img_idx.tolist(), [0,3,6,9])
        for ids, scales in (([3],[1]),([0,0],[1,1]),([0],[]),([0],[0])):
            with self.assertRaises(ValueError): fn(source,ids,scales)

    def test_legacy_pbr_requires_compatible_light_and_background(self):
        import runpy
        check = runpy.run_path(str(REPO/'sim_render/checkpoint_contract.py'))['require_supported_scene_features']
        check({'Background': {'ctrl': {'sh_degree': 3}}})
        check({'Background': {'type': 'models.gaussians.VanillaGaussians', 'ctrl': {'use_pbr': True}},
               'GlobalLighting': {'type': 'models.modules.GlobalLightingModule'}})
        for config in ({'Background': {'ctrl': {'use_pbr': True}}},
                       {'GlobalLighting': {'type': 'other.Light'}}):
            with self.assertRaises(ValueError): check(config)

    def test_online_requires_extrinsic_and_uses_it(self):
        fn=extract('datasets/base/pixel_source.py','ScenePixelSource','prepare_online_render_data',dict(torch=torch))
        source=SimpleNamespace(_build_online_render_data=lambda **kw: kw)
        pose=torch.eye(4)[None]
        with self.assertRaises(ValueError): fn(source,'waymo',pose,SimpleNamespace(),0)
        extrinsic=np.eye(4); extrinsic[:3,3]=[1,2,3]
        result=fn(source,'waymo',pose,SimpleNamespace(cam_to_ego=extrinsic),0)
        np.testing.assert_allclose(result['traj'][0],extrinsic)

    def test_online_embedding_and_single_frame_time(self):
        def rays(x,y,c2w,k):
            return torch.zeros(len(x),3),torch.ones(len(x),3),torch.ones(len(x),1)
        fn=extract('datasets/base/pixel_source.py','ScenePixelSource','_build_online_render_data',dict(torch=torch,np=np,get_rays=rays))
        camera=SimpleNamespace(intrinsics=torch.eye(3)[None],distortions=torch.zeros(1,5),
            HEIGHT=2,WIDTH=2,unique_img_idx=torch.tensor([2]),egocar_mask=None)
        source=SimpleNamespace(device='cpu',num_frames=1)
        result=fn(source,'waymo',[torch.eye(4)],camera,0)[0]['image_infos']
        self.assertTrue(torch.all(result['img_idx']==2))
        self.assertEqual(result['normed_time'].item(),0)
        for idx in (None,-1,1,.5,True):
            with self.assertRaises(ValueError): fn(source,'waymo',[torch.eye(4)],camera,idx)

if __name__=='__main__': unittest.main()
