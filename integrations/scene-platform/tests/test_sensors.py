"""Analytic geometry and actual recorder tests, not a GPU-scene certification."""

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

import copy,sys,tempfile,types,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sensor_contracts import configuration
from optional_sensors import beams,face_rotation,FACE_DIRECTIONS,sample_cube,rgbd_cloud,Recorder,render_cube
from platform_contracts import compile_task
from platform_catalog import Catalog
from platform_io import LOCAL,read

def cube(depth=10.,size=128):
    k=np.array([[size/2,0,(size-1)/2],[0,size/2,(size-1)/2],[0,0,1.]])
    return [dict(depth=np.full((size,size),depth),depth_alpha=np.ones((size,size)),K=k) for _ in range(6)]

class Sensors(unittest.TestCase):
    def test_default_disabled_has_no_render_or_files(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'sensor';r=Recorder(None,out)
            self.assertIsNone(r.record(None,{}));r.flush();self.assertFalse(out.exists())
    def test_reject_unknown_provider_and_bad_calibration(self):
        for v in [dict(provider='physical_lidar'),dict(enabled=1),dict(channels=True),dict(max_range_m=float('nan')),dict(sensor_to_ego=np.zeros((4,4)).tolist()),dict(online_control=True)]:
            with self.subTest(value=v),self.assertRaises(ValueError):configuration(v)
    def test_virtual_sensor_is_bound_to_v2_only_and_compiled(self):
        task=read(LOCAL/'examples/cutin_30.json');task['sensors']=dict(lidar=dict(enabled=True))
        c=compile_task(task,Catalog());self.assertTrue(any(m['module']=='optional_pointcloud_recorder' for m in c['module_calls']))
        task['sensors']['radar']={}
        with self.assertRaises(ValueError):compile_task(task,Catalog())
    def test_six_face_distance_rays_and_handedness(self):
        c=configuration(dict(enabled=True,channels=3,azimuth_samples=32,elevation_min_deg=-20,elevation_max_deg=20))
        for d in FACE_DIRECTIONS:
            rot=face_rotation(d);np.testing.assert_allclose(rot.T@rot,np.eye(3));self.assertAlmostEqual(np.linalg.det(rot),1.)
        points=sample_cube(cube(),c);rays=beams(c)
        expected=10/np.max(np.abs(rays),axis=1)
        np.testing.assert_allclose(points['range_m'],expected,rtol=1e-6)
        self.assertTrue(points['valid'].all());np.testing.assert_allclose(np.linalg.norm(points['xyz_sensor'],axis=1),expected,rtol=1e-6)
    def test_missing_low_alpha_and_out_of_range_are_not_hits(self):
        c=configuration(dict(enabled=True));views=cube();views[0]['depth_alpha'][:]=0
        data=sample_cube(views,c);f=np.argmax(beams(c)@FACE_DIRECTIONS.T,axis=1)
        self.assertFalse(data['valid'][f==0].any())
        self.assertFalse(sample_cube(cube(300),c)['valid'].any())
        with self.assertRaises(ValueError):sample_cube(views[:3],c)
    def packet(self,depth=10.,frame=10):
        pose=np.eye(4);pose[0,3]=20
        k=np.array([[10,0,1],[0,10,1],[0,0,1.]])
        return dict(source_frame=frame,timestamp_s=frame/10,ego_to_world=pose,camera_order=['front'],
            cameras=dict(front=dict(depth=np.full((3,3),depth),depth_alpha=np.ones((3,3)),K=k,
                camera_to_world=pose,depth_encoding='expected_camera_z_v1')),
            model_cache_token='generation:1')
    def test_calibrated_rgbd_mount_and_range(self):
        c=configuration(dict(provider='camera_depth_pointcloud_v1',camera_pixel_stride=1))
        p=self.packet();data=rgbd_cloud(p,c);self.assertEqual(len(data['xyz_sensor']),9)
        np.testing.assert_allclose(data['xyz_sensor'][4],[0,0,8.2],atol=1e-6)
        p['cameras']['front']['depth_encoding']='framework_ray_depth'
        with self.assertRaises(ValueError):rgbd_cloud(p,c)
    def test_edit_changes_geometry_not_metadata_and_reset_restores(self):
        c=configuration(dict(provider='camera_depth_pointcloud_v1',camera_pixel_stride=1))
        original=rgbd_cloud(self.packet(10),c);edited=rgbd_cloud(self.packet(5),c);reset=rgbd_cloud(self.packet(10),c)
        self.assertFalse(np.allclose(original['xyz_sensor'],edited['xyz_sensor']))
        np.testing.assert_array_equal(original['xyz_sensor'],reset['xyz_sensor'])
    def test_real_npz_ply_receipt_clock_and_no_controller_consumer(self):
        with tempfile.TemporaryDirectory() as d:
            session=types.SimpleNamespace(spec=types.SimpleNamespace(fps=10),generation='generation',revision=1,backend=types.SimpleNamespace())
            recorder=Recorder(dict(enabled=True,provider='camera_depth_pointcloud_v1',camera_pixel_stride=1),d)
            r=recorder.record(session,self.packet());self.assertFalse(r['feeds_controller']);self.assertFalse(r['feeds_UniAD'])
            self.assertEqual(r['extra_gaussian_view_renders'],0)
            with np.load(r['points']['path']) as z:
                self.assertEqual(z['source_timestamp_s'],1.);self.assertEqual(z['local_timestamp_s'],0.);self.assertEqual(z['xyz_sensor'].shape,(9,3))
            self.assertIn(b'element vertex 9',Path(r['ply']['path']).read_bytes())
            self.assertEqual(read(Path(d)/'manifest.json')['frame_count'],1)
            with self.assertRaises(FileExistsError):recorder.record(session,self.packet())
    def test_cube_renderer_restores_backend_even_on_error(self):
        import torch
        camera=types.SimpleNamespace(intrinsics=torch.eye(3)[None],distortions=None,unique_img_idx=None)
        backend=types.SimpleNamespace(torch=torch,frames=2,cameras_data={0:camera,1:camera,2:camera},_last='saved')
        original=backend.cameras_data
        def fail(*args):raise RuntimeError('render fault')
        backend.render=fail
        with self.assertRaisesRegex(RuntimeError,'render fault'):render_cube(backend,np.eye(4),0,configuration())
        self.assertIs(backend.cameras_data,original);self.assertEqual(backend._last,'saved')
    def test_cube_renderer_covers_six_orientations_in_fixed_order(self):
        import torch
        camera=types.SimpleNamespace(intrinsics=torch.eye(3)[None],distortions=None,unique_img_idx=None)
        backend=types.SimpleNamespace(torch=torch,frames=2,cameras_data={2:camera,0:camera,1:camera},_last='saved')
        original=backend.cameras_data;directions=[]
        def render(pose,t):
            views={}
            for i,name in enumerate(['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']):
                proxy=backend.cameras_data[i];directions.append(proxy.cam_to_ego[:3,2])
                n=proxy.HEIGHT;views[name]=dict(depth=np.full((n,n),10.),depth_alpha=np.ones((n,n)),K=proxy.intrinsics[0].numpy(),depth_encoding='expected_camera_z_v1')
            backend._last='changed';return dict(cameras=views)
        backend.render=render
        result=render_cube(backend,np.eye(4),0,configuration())
        self.assertEqual(len(result),6);np.testing.assert_allclose(directions,FACE_DIRECTIONS)
        self.assertIs(backend.cameras_data,original);self.assertEqual(backend._last,'saved')

if __name__=='__main__':unittest.main()
