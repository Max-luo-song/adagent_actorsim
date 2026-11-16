
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

import json
pass  # Path is provided by the portable boundary above
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scene_api import SceneSession,SceneSpec,EditRequest,BASE,CAMERA_ORDER
from asset_calibration import RULE,fingerprint
from asset_geometry import SCHEMA,declared_contract


class FakeBackend:
    def __init__(self,spec):
        self.spec=spec;self.assets={};self.timestamp=0.;self.speed=0.;self.steer=0.;self.x=0.
        self.closed=False;self.failure=False;self.bad_depth=False
    def apply(self,edit):
        if self.failure:raise RuntimeError('Injected allocation failure')
        if edit.operation=='remove':del self.assets[edit.object_id]
        else:self.assets[edit.object_id]=edit.asset_to_world.copy()
    def render(self,pose,time):
        cameras={}
        for cid,name in enumerate(self.spec.camera_order):
            cameras[name]=dict(rgb=np.zeros((3,4,3),np.uint8),depth=np.full((3,4),float('nan') if self.bad_depth else 1.,np.float32),
                K=np.eye(3),camera_to_world=pose.copy(),source_camera_id=cid)
        return dict(cameras=cameras,ego_state={'speed_mps':self.speed})
    def state(self):return dict(timestamp=self.timestamp,speed=self.speed,steer=self.steer)
    def current_view(self):
        pose=np.eye(4);pose[0,3]=self.x
        return self.timestamp,pose
    def advance(self,action):
        dt=self.spec.vehicle.dt;self.timestamp+=dt;self.speed+=action['acc_mps2']*dt
        self.x+=self.speed*dt;self.steer+=action['steer_rate_radps']*dt
    def reset(self):self.assets.clear();self.timestamp=self.speed=self.steer=self.x=0.
    def close(self):self.closed=True


class SceneAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);self.asset=root/'asset';self.asset.mkdir()
        (self.asset/'gs.pth').write_bytes(b'contract fixture, not a render model')
        (self.asset/'wlh.json').write_text('[1.8,4,1]')
        self.calibration=root/'calibration.json'
        self.calibration.write_text(json.dumps(dict(schema='gaussian_center_support_calibration_v1',geometry_schema=SCHEMA,
            rule=RULE,asset=str(self.asset),source_files={name:fingerprint(self.asset/name) for name in ('gs.pth','wlh.json')},
            support=dict(bounds_raw=[[-2,-1,-.9],[2,0,.9]],wlh=[1.8,4,1],raw_recenter_xz=[0,0],bottom_y=0),
            geometry_contract=declared_contract([1.8,4,1],0),independent_physical_calibration=False,
            n1_certified=False,prediction_data_used=False,scene_data_used=False)))
        self.session=SceneSession(FakeBackend)
        self.manifests=[BASE/f'integrations/sr-hugsim/configs/waymo{scene}.json' for scene in (172,552)]
        self.session.load_scene(self.manifests[0])
    def tearDown(self):self.session.close();self.tmp.cleanup()
    def request(self):return dict(operation='insert',object_id='car1',mode='diagnostic_preview',
        asset=str(self.asset),asset_calibration=str(self.calibration),asset_to_world=np.eye(4).tolist())
    def test_two_real_scene_manifests_use_same_contract(self):
        for manifest,scene in zip(self.manifests,(172,552)):
            spec=SceneSpec.read(manifest);self.assertEqual(spec.source['scene'],scene)
            self.assertEqual(spec.camera_order,CAMERA_ORDER)
            self.session.load_scene(manifest)
            packet=self.session.observe();self.assertEqual(packet['scene'],scene)
            self.assertEqual(packet['camera_order'],list(CAMERA_ORDER))
            self.assertFalse(packet['formal_evaluation_admitted'])
    def test_insert_move_remove_revisions_and_no_input_alias(self):
        request=self.request();self.session.edit(request);request['asset_to_world'][0][3]=999
        self.assertEqual(self.session.objects['car1']['asset_to_world'][0][3],0)
        moved=np.eye(4);moved[1,3]=2
        self.session.edit(dict(operation='set_pose',object_id='car1',mode='diagnostic_preview',asset_to_world=moved.tolist()))
        self.assertEqual(self.session.observe()['edit_revision'],2)
        self.session.edit(dict(operation='remove',object_id='car1',mode='diagnostic_preview'))
        self.assertEqual(self.session.objects,{})
        self.assertEqual(self.session.revision,3)
    def test_duplicate_missing_id_and_formal_mode_rejected(self):
        self.session.edit(self.request())
        with self.assertRaises(ValueError):self.session.edit(self.request())
        with self.assertRaises(KeyError):self.session.edit(dict(operation='remove',object_id='missing',mode='diagnostic_preview'))
        with self.assertRaises(ValueError):self.session.edit(dict(self.request(),mode='formal'))
        self.assertEqual(self.session.revision,1)
    def test_backend_failure_does_not_commit_edit(self):
        self.session.backend.failure=True
        with self.assertRaises(RuntimeError):self.session.edit(self.request())
        self.assertEqual(self.session.revision,0);self.assertEqual(self.session.objects,{})
    def test_reset_and_scene_switch_clear_objects_and_history(self):
        self.session.edit(self.request());first=self.session.generation
        self.session.step(dict(acc_mps2=1.,steer_rate_radps=0.))
        packet=self.session.reset();self.assertNotEqual(packet['history_token'],first)
        self.assertEqual(packet['managed_objects'],{});self.assertEqual(packet['timestamp_s'],0.)
        self.session.edit(self.request());old=self.session.backend;token=self.session.generation
        self.session.load_scene(self.manifests[1])
        self.assertTrue(old.closed);self.assertEqual(self.session.objects,{})
        self.assertNotEqual(self.session.generation,token)
    def test_invalid_pose_and_asset_identity_fail_before_mutation(self):
        pose=np.eye(4);pose[0,0]=-1
        with self.assertRaises(ValueError):self.session.edit(dict(self.request(),asset_to_world=pose.tolist()))
        (self.asset/'gs.pth').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.session.edit(self.request())
        self.assertEqual(self.session.revision,0)
    def test_render_is_stateless_clock_and_action_guards(self):
        before=self.session.backend.state();pose=np.eye(4);pose[:3,3]=[1,2,3]
        packet=self.session.render(pose,.5)
        self.assertEqual(packet['source_frame'],5);self.assertEqual(self.session.backend.state(),before)
        np.testing.assert_allclose(packet['cameras']['CAM_FRONT']['camera_to_ego'],np.eye(4))
        for timestamp in (-1,float('nan'),self.session.spec.frames/self.session.spec.fps):
            with self.assertRaises(ValueError):self.session.render(pose,timestamp)
        for action in (dict(acc_mps2=100.,steer_rate_radps=0.),dict(acc_mps2=-1.,steer_rate_radps=0.),dict(acc=1,steer=0)):
            with self.assertRaises(ValueError):self.session.step(action)
        self.assertEqual(self.session.backend.state(),before)
    def test_bad_packet_and_failed_manifest_do_not_close_old_scene(self):
        self.session.backend.bad_depth=True
        with self.assertRaises(ValueError):self.session.observe()
        manifest=_hugsim_expand(json.loads(self.manifests[0].read_text()));manifest['camera_order']=['CAM_FRONT']*3
        path=Path(self.tmp.name)/'bad.json';path.write_text(json.dumps(manifest));old=self.session.backend
        with self.assertRaises(ValueError):self.session.load_scene(path)
        self.assertIs(self.session.backend,old);self.assertFalse(old.closed)
    def test_native_render_failure_restores_process_and_camera_state(self):
        import torch
        from native_scene_backend import NativeSceneBackend,Asset
        target=NativeSceneBackend.__new__(NativeSceneBackend)
        target.torch=torch
        native_pose=np.eye(4);native_pose[0,3]=2
        old_pose=lambda:native_pose.copy()
        camera=SimpleNamespace(dynamics={'existing':torch.eye(4)})
        old_dynamics=camera.dynamics
        pc=SimpleNamespace(active_sh_degree=0,_xyz=torch.zeros(2,3),
            _features_dc=torch.zeros(2,1,3),_features_rest=torch.empty(2,0,3))
        old_rest=pc._features_rest
        def fail(*args,**kwargs):
            self.assertIn('unified_preview_car',kwargs['dynamic_gaussians'])
            self.assertEqual(pc.active_sh_degree,3)
            raise RuntimeError('Rasterization failed')
        target.renderer=SimpleNamespace(render=fail)
        target.assets={'car':Asset(object(),None,np.eye(4),torch.eye(4),Path('asset'))}
        target.env=SimpleNamespace(ego_pose=old_pose,motion=SimpleNamespace(timestamp=.3,speed=0.,steer=0.))
        target.env.observe=lambda:target.renderer.render(viewpoint=camera,pc=pc,dynamic_gaussians={})
        with self.assertRaisesRegex(RuntimeError,'Rasterization failed'):
            target.render(np.eye(4),.7)
        self.assertIs(target.renderer.render,fail)
        self.assertIs(target.env.ego_pose,old_pose)
        self.assertEqual(target.env.motion.timestamp,.3)
        self.assertEqual(pc.active_sh_degree,0)
        self.assertIs(pc._features_rest,old_rest)
        self.assertIs(camera.dynamics,old_dynamics)

    def test_native_constructor_failure_releases_single_session_slot(self):
        from unittest.mock import patch
        import native_scene_backend as module
        import os
        cwd=os.getcwd()
        spec=SceneSpec.read(self.manifests[0])
        with patch('torch.cuda.is_available',return_value=False):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError,'CUDA GPU required'):
                    module.NativeSceneBackend(spec)
                self.assertIsNone(module._ACTIVE)
                self.assertEqual(os.getcwd(),cwd)

    def test_failed_backend_switch_leaves_session_unloaded(self):
        old=self.session.backend
        def fail(spec):raise RuntimeError('No GPU')
        self.session.factory=fail
        with self.assertRaises(RuntimeError):self.session.load_scene(self.manifests[1])
        self.assertTrue(old.closed);self.assertIsNone(self.session.backend)
        with self.assertRaises(RuntimeError):self.session.observe()

if __name__=='__main__':unittest.main()
