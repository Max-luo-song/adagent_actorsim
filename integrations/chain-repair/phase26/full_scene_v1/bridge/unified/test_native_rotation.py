"""CPU rotation contract, real renderer transforms, proxy and reset evidence."""

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
import json
pass  # Path is provided by the portable boundary above
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'weather_runtime'))
for parent in HERE.parents:
    if (parent/'integrations/sr-hugsim').is_dir():
        sys.path.insert(0, str(parent/'integrations/sr-hugsim'))
        break
from edit_api import SceneEditor, read_command, native_request, UnsupportedEditError
from native_edit import rotate_native
from native_rotation_probe import probe_native_rotation
from runtime_contract import CAMERAS
from scene_edit_session import UnifiedSceneSession, saf


def command(yaw=15., **extra):
    return dict(schema='scene_edit_v1', mode='diagnostic_preview', operation='rotate',
                object_id='native:RigidNodes:1', yaw_degrees=yaw, **extra)


def matrices(q):
    return Rotation.from_quat(q.detach().numpy()[..., [1,2,3,0]]).as_matrix()


class RotationTests(unittest.TestCase):
    def setUp(self):
        q = Rotation.from_euler('xyz', np.arange(90*2*3).reshape(-1,3)*.007).as_quat()[:,[3,0,1,2]]
        self.model = SimpleNamespace(instances_trans=torch.arange(540,dtype=torch.float64).reshape(90,2,3)/100,
            instances_quats=torch.tensor(q.reshape(90,2,4)),
            _means=torch.tensor([[1.,2.,3.]],dtype=torch.float64),
            _quats=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64),
            _scales=torch.tensor([[1.,2.,3.]],dtype=torch.float64),
            instances_size=torch.tensor([[4.,2.,1.],[4.,2.,1.]],dtype=torch.float64),
            instances_fv=torch.ones(90,2,dtype=torch.bool))
        self.backend = SimpleNamespace(torch=torch, _last=None,
            trainer=SimpleNamespace(models={'RigidNodes':self.model}),
            env=SimpleNamespace(models={'RigidNodes':{k:getattr(self.model,k) for k in
                ('instances_trans','instances_quats','instances_size','instances_fv')}}))
        self.backend.reset=lambda:setattr(self.backend,'_last',None)
        self.session=SimpleNamespace(backend=self.backend,generation='A',revision=0,objects={})
        self.editor=SceneEditor(self.session,lambda c:None)
        self.original_q=self.model.instances_quats.clone()
        self.original_t=self.model.instances_trans.clone()

    def test_world_left_product_preserves_centers_other_instances_and_local_covariance(self):
        self.editor.edit(command())
        expected=Rotation.from_euler('z',15,degrees=True).as_matrix() @ matrices(self.original_q[:,1])
        np.testing.assert_allclose(matrices(self.model.instances_quats[:,1]),expected,atol=1e-12)
        self.assertTrue(torch.equal(self.model.instances_quats[:,0],self.original_q[:,0]))
        self.assertTrue(torch.equal(self.model.instances_trans,self.original_t))
        self.assertIs(self.backend.env.models['RigidNodes']['instances_quats'],self.model.instances_quats)
        self.assertEqual(self.editor.objects()[1]['yaw_world_degrees'],15.)
        self.assertEqual(self.editor.capabilities()['native_rotate_groups'],['RigidNodes'])

    def test_real_renderer_mean_and_covariance_transforms_rotate_about_root(self):
        # Execute the actual renderer methods, without loading GPU/training imports.
        path=HERE.parent.parent/'framework_repo/models/nodes/rigid.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(
            isinstance(f,ast.FunctionDef) and f.name=='transform_means' for f in n.body))
        funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('transform_means','transform_quats')]
        def rot(q):return torch.tensor(matrices(q),dtype=q.dtype)
        def mult(a,b):
            out=(Rotation.from_quat(a.numpy()[:,[1,2,3,0]])*Rotation.from_quat(b.numpy()[:,[1,2,3,0]])).as_quat()
            return torch.tensor(out[:,[3,0,1,2]],dtype=a.dtype)
        scope=dict(torch=torch,quat_to_rotmat=rot,quat_mult=mult)
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),scope)
        m=self.model;m.in_test_set=False;m.cur_frame=80;m.point_ids=torch.tensor([[1]])
        m.quat_act=lambda q:q/q.norm(dim=-1,keepdim=True)
        means0=scope['transform_means'](m,m._means).numpy()
        quats0=scope['transform_quats'](m,m._quats)
        self.editor.edit(command())
        means1=scope['transform_means'](m,m._means).numpy()
        quats1=scope['transform_quats'](m,m._quats)
        delta=Rotation.from_euler('z',15,degrees=True).as_matrix();center=m.instances_trans[80,1].numpy()
        np.testing.assert_allclose(means1,(means0-center)@delta.T+center,atol=2e-6)
        r0,r1=matrices(quats0),matrices(quats1)
        cov0=r0@np.diag([1.,4.,9.])@r0.transpose(0,2,1)
        cov1=r1@np.diag([1.,4.,9.])@r1.transpose(0,2,1)
        np.testing.assert_allclose(cov1,delta@cov0@delta.T,atol=1e-12)

    def test_composed_edits_and_reset_exactly_restore_both_tensors(self):
        self.editor.edit(command(30))
        self.editor.edit(dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='translate',
            object_id='native:RigidNodes:1',offset_world_m=[.5,0,0]))
        self.editor.edit(command(-15,expected_revision=2,expected_generation='A'))
        self.assertEqual(self.editor.yaws[('RigidNodes',1)],15.)
        self.editor.restore_native()
        self.assertTrue(torch.equal(self.model.instances_quats,self.original_q))
        self.assertTrue(torch.equal(self.model.instances_trans,self.original_t))

    def test_history_failure_rolls_back_without_revision_or_journal(self):
        def fail():raise RuntimeError('history failure')
        self.backend.clear_observation_history=fail
        with self.assertRaisesRegex(RuntimeError,'history'):self.editor.edit(command())
        self.assertTrue(torch.equal(self.model.instances_quats,self.original_q))
        self.assertFalse(self.editor.original_quaternions)
        self.assertFalse(self.editor.journal)
        self.assertEqual(self.session.revision,0)

    def test_invalid_values_and_unsupported_groups_leave_geometry_unchanged(self):
        for yaw in (True,float('nan'),float('inf'),46,'15',None):
            with self.subTest(yaw=yaw),self.assertRaises(ValueError):self.editor.edit(command(yaw))
        for group in ('SMPLNodes','DeformableNodes'):
            with self.assertRaises(UnsupportedEditError):read_command(dict(command(),object_id=f'native:{group}:1'))
        for extra in ({'expected_revision':1},{'expected_generation':'B'}):
            with self.assertRaises(ValueError):self.editor.edit(command(**extra))
        self.assertTrue(torch.equal(self.model.instances_quats,self.original_q))

    def test_cumulative_limit_and_runtime_guard(self):
        self.editor.edit(command(30))
        with self.assertRaisesRegex(ValueError,'cumulative'):self.editor.edit(command(16))
        self.editor.started=True
        with self.assertRaisesRegex(RuntimeError,'pre-run'):self.editor.edit(command(-15))
        self.assertEqual(self.session.revision,1)

    def test_proxy_identity_and_invalid_source_quaternions_are_atomic(self):
        self.backend.env.models['RigidNodes']['instances_quats']=self.original_q.clone()
        with self.assertRaisesRegex(RuntimeError,'identity'):self.editor.edit(command())
        self.backend.env.models['RigidNodes']['instances_quats']=self.model.instances_quats
        self.model.instances_quats[0,1]=0
        with self.assertRaisesRegex(ValueError,'source'):self.editor.edit(command())
        self.assertEqual(self.session.revision,0)

    def test_restore_validates_all_shared_tensors_before_any_mutation(self):
        self.editor.edit(command())
        rotated=self.model.instances_quats.clone()
        self.backend.env.models['RigidNodes']['instances_quats']=rotated.clone()
        with self.assertRaises(RuntimeError):self.editor.restore_native()
        self.assertTrue(torch.equal(self.model.instances_quats,rotated))

    def test_session_packet_and_reset_clear_rotation(self):
        spec=SimpleNamespace(source={'scene':552},fps=10.,frames=90,
            camera_order=CAMERAS,summary=lambda:dict(backend='legacy',full_sr_trainer=False))
        self.backend.spec=spec;self.backend.close=lambda:None
        def render(pose,timestamp):
            self.backend._last=(timestamp,pose)
            c=dict(rgb=np.zeros((4,6,3),np.uint8),depth=np.ones((4,6),np.float32),
                K=np.eye(3),camera_to_world=pose,source_camera_id=0)
            return dict(cameras={name:dict(c) for name in CAMERAS},ego_state={})
        self.backend.render=render
        with patch.object(saf.SceneSpec,'read',return_value=spec):
            s=UnifiedSceneSession(lambda spec:self.backend);s.load_scene('fake')
            s.edit(command());packet=s.render(np.eye(4),8.)
            self.assertTrue(packet['edit_trust']['edited'])
            self.assertEqual(packet['native_edits'][0]['yaw_degrees'],15.)
            s.reset()
            self.assertEqual(s.revision,0);self.assertFalse(s.editor.yaws)
            self.assertTrue(torch.equal(self.model.instances_quats,self.original_q))
            self.assertFalse(s.render(np.eye(4),8.)['edit_trust']['edited'])
            s.close()

    def probe(self,folder,corrupt=False,hidden=False):
        (folder/'ego_pose').mkdir()
        for frame in (0,80):np.savetxt(folder/f'ego_pose/{frame:03d}.txt',np.eye(4))
        self.backend.spec=SimpleNamespace(source=dict(inputs=str(folder),start_timestep=0))
        def render(backend,pose,t):
            moved=not hidden and not torch.equal(self.original_q,self.model.instances_quats)
            rgb=np.zeros((4,6,3),np.uint8);mask=np.zeros((4,6),np.uint8)
            rgb[1,1+int(moved)]=100;mask[1,1+int(moved)]=255
            backend._last=(t,pose)
            return dict(cameras={name:dict(rgb=rgb,weather_dynamic_mask=mask,K=np.eye(3),camera_to_world=pose) for name in CAMERAS})
        def apply(edit):
            receipt=self.editor.edit(command())
            if corrupt:self.model._means.add_(1)
            return receipt
        return probe_native_rotation(self.backend,native_request(command()),folder/'probe',render,apply)

    def test_rotation_probe_checks_actual_oriented_proxy_and_discarded_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=self.probe(Path(tmp))
            self.assertTrue(report['passed']);self.assertTrue(report['oriented_proxy_matches_rotation'])
            self.assertIsNone(self.backend._last)
            self.assertEqual(len(self.editor.journal),1)

    def test_rotation_probe_rejects_local_gaussian_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'probe failed'):self.probe(Path(tmp),corrupt=True)
            self.assertFalse(_hugsim_expand(json.loads((Path(tmp)/'probe/report.json').read_text()))['local_gaussian_geometry_unchanged'])

    def test_rotation_probe_rejects_invisible_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'probe failed'):self.probe(Path(tmp),hidden=True)


if __name__=='__main__':unittest.main()
