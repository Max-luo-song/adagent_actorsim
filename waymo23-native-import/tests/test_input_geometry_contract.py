"""Execute the parser function on CPU and check non-default image sizes."""
import ast
import copy
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from input_geometry import camera_projection, intrinsic_matrix, model_input_audit, resize_plan
from edit_trust import assert_edit_evaluable


def cpu_parser(path=None):
    import torch
    path = path or BASE/'code/UniAD_SIM/tools/closeloop/dataparser.py'
    tree = ast.parse(path.read_text())
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_raw'))
    # Inject the same NumPy-only gate without importing the GPU plugin package.
    function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]
    scope = dict(np=np, torch=torch, SCR=Rotation, math=math, assert_edit_evaluable=assert_edit_evaluable,
        camera_projection=camera_projection, get_intrinsic=intrinsic_matrix, resize_plan=resize_plan,
        OPENCV2IMU=np.array([[0,0,1],[-1,0,0],[0,-1,0]]),
        OPENCV2LIDAR=np.array([[1,0,0],[0,0,1],[0,-1,0]]),
        cv2=SimpleNamespace(resize=lambda image, wh: np.zeros((wh[1],wh[0],3), dtype=image.dtype)),
        mmcv=SimpleNamespace(imnormalize=lambda image, mean, std, to_rgb: (image.astype(np.float32)-mean)/std))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), scope)
    return scope['parse_raw']


class InputGeometryContractTests(unittest.TestCase):
    def test_actual_parser_matches_metadata_projection_for_unseen_resolutions(self):
        import torch
        parse = cpu_parser()
        for width, height in ((960,640), (1234,789), (640,480), (1920,1280)):
            intrinsic = dict(W=width,H=height,cx=width*.47,cy=height*.52,fovx=.9,fovy=.65)
            camera = dict(intrinsic=intrinsic,l2c=np.eye(4))
            rgb = np.zeros((height,width,3), dtype=np.uint8)
            info = dict(cam_params=dict(CAM_FRONT=camera), ego_rot=[0,0,0],ego_pos=[0,0,0],
                        timestamp=1.,accelerate=0.,ego_velo=2.,ego_steer=0.,steer_rate=0.,command=2)
            with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                data = parse((dict(rgb=dict(CAM_FRONT=rgb)),info), ['CAM_FRONT'],
                             dict(mean=np.zeros(3),std=np.ones(3),to_rgb=False), None)
            k, p = camera_projection(camera, rgb.shape)
            np.testing.assert_allclose(data['img_metas'][0][0]['intrinsics'][0], k)
            np.testing.assert_allclose(data['img_metas'][0][0]['lidar2img'][0], p)
            self.assertEqual(tuple(data['img'].shape[-2:]), (900,1600))
            self.assertEqual(json.loads(json.dumps(model_input_audit(data)))['stage'], 'before_model_forward')

    def test_resize_projection_commutes_with_pixel_scaling(self):
        rng = np.random.default_rng(7)
        points = np.column_stack([rng.uniform(-2,2,(40,2)),rng.uniform(3,30,40),np.ones(40)])
        for width,height in ((901,613),(1527,977),(1920,1080)):
            intrinsic=dict(W=width,H=height,cx=width*.46,cy=height*.53,fovx=1.1,fovy=.8)
            original=intrinsic_matrix(intrinsic)
            resized=intrinsic_matrix(intrinsic,(1600,900))
            a=points@original.T;b=points@resized.T
            np.testing.assert_allclose(a[:,:2]/a[:,2:3]*[1600/width,900/height],b[:,:2]/b[:,2:3],atol=1e-10)

    def test_letterbox_projection_includes_padding_and_exact_rounded_resize(self):
        for width,height in ((960,640),(641,479),(300,1000)):
            intr=dict(W=width,H=height,cx=width*.47,cy=height*.52,fovx=1.,fovy=.8)
            plan=resize_plan((width,height),mode='letterbox')
            original=intrinsic_matrix(intr)
            transformed=intrinsic_matrix(intr,(1600,900),resize_mode='letterbox')
            np.testing.assert_allclose(transformed,plan['affine']@original)
            left,top,right,bottom=plan['pad_ltrb']
            rw,rh=plan['resized_wh']
            self.assertEqual(rw+left+right,1600)
            self.assertEqual(rh+top+bottom,900)
            self.assertLessEqual(abs(rw/width-rh/height),1/min(width,height))
        with self.assertRaises(ValueError):resize_plan((960,640),mode='unknown')

    def test_actual_parser_letterbox_normalized_padding_and_geometry(self):
        import torch
        parse=cpu_parser()
        width,height=960,640
        camera=dict(intrinsic=dict(W=width,H=height,cx=480,cy=320,fovx=1.,fovy=.8),l2c=np.eye(4))
        rgb=np.zeros((height,width,3),dtype=np.uint8)
        info=dict(cam_params=dict(CAM_FRONT=camera),ego_rot=[0,0,0],ego_pos=[0,0,0],timestamp=1.,
                  accelerate=0.,ego_velo=2.,ego_steer=0.,steer_rate=0.,command=2)
        with patch.object(torch.Tensor,'cuda',lambda tensor:tensor):
            data=parse((dict(rgb=dict(CAM_FRONT=rgb)),info),['CAM_FRONT'],
                       dict(mean=np.ones(3)*10,std=np.ones(3),to_rgb=False),None,resize_mode='letterbox')
        image=data['img'].numpy()[0,0]
        self.assertEqual(image.shape,(3,900,1600))
        self.assertTrue((image[:,:,:125]==0).all())
        self.assertTrue((image[:,:,125:1475]==-10).all())
        k,p=camera_projection(camera,rgb.shape,resize_mode='letterbox')
        np.testing.assert_allclose(data['img_metas'][0][0]['lidar2img'][0],p)
        self.assertAlmostEqual(k[0,2],800.)

    def test_bad_calibration_and_wrong_source_size_fail_before_model(self):
        intr=dict(W=960,H=640,cx=480,cy=320,fovx=1.,fovy=.8)
        with self.assertRaisesRegex(ValueError,'size mismatch'):
            camera_projection(dict(intrinsic=intr,l2c=np.eye(4)),(900,1600,3))
        invalid=np.eye(4);invalid[0,0]=2.
        with self.assertRaises(ValueError):
            camera_projection(dict(intrinsic=intr,l2c=invalid),(640,960,3))
