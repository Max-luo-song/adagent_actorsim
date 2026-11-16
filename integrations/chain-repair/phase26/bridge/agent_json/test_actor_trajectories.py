"""Real trajectory/editor code with an explicit CPU body/raster test double."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import copy,json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_agent_bridge as fixtures
from actor_trajectories import source_center,compile_path,sample_path,ground_basis
from agent_jobs import run_job
import agent_bootstrap as bootstrap


def spherical_harmonics(degree,directions,coefficients):
    # Deliberately directional CPU evaluator for checking inverse view rotation.
    return coefficients[:,0]+(directions[...,None]*coefficients[:,1:4]).sum(1)


class Body:
    def __init__(self):
        self.instances_trans=torch.zeros(150,2,3)
        for f in range(150):self.instances_trans[f,:,0]=f*.01
        # A nonlinear heldout root exposes accidental source-root interpolation.
        self.instances_trans[80,:,0]=1.7
        self.instances_quats=torch.zeros(150,2,1,4);self.instances_quats[...,0]=1
        self.instances_fv=torch.ones(150,2,dtype=torch.bool)
        self.point_ids=torch.tensor([[0],[0],[0],[1],[1],[1]])
        self._means=torch.tensor([[0.,0.,0.],[.1,0.,1.],[-.2,.1,.5]]*2)
        self._quats=torch.zeros(6,4);self._quats[:,0]=1
        self._features_dc=torch.full((6,3),.05);self._features_rest=torch.full((6,3,3),.03)
        self.smpl_qauts=torch.zeros(150,2,23,4);self.smpl_qauts[...,0]=1
        self.sh_degree=1;self.step=100;self.ctrl_cfg=SimpleNamespace(sh_degree_interval=1)
        self.cur_frame=0;self.fail=False
    def get_gaussians(self,cam):
        if self.fail:raise RuntimeError('CPU body failure')
        ids=self.point_ids.flatten();f=self.cur_frame
        root=self.instances_trans[f].clone()
        if f in (80,90,100,110,120):root=(self.instances_trans[f-1]+self.instances_trans[f+1])*.5
        means=self._means+root[ids]
        means=means.clone();means[1::3,1]+=f*.0001 # source-clock articulation
        coeff=torch.cat([self._features_dc[:,None],self._features_rest],1)
        dirs=torch.nn.functional.normalize(means-cam.camtoworlds[:3,3],dim=-1)
        rgb=(spherical_harmonics(1,dirs,coeff)+.5).clamp(0,1)
        self.filter_mask=torch.ones(6,dtype=torch.bool)
        return dict(_means=means,_quats=self._quats.clone(),_rgbs=rgb,_scales=torch.ones(6,3)*.1,_opacities=torch.ones(6,1))


class ActorTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.BridgeTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.b=self.f.bridge;self.root=self.f.root
        from scene_edit_session import saf
        old=saf.SceneSpec.read.side_effect
        def read(path):
            spec=old(path);spec.frames=150;return spec
        saf.SceneSpec.read.side_effect=read
        for frame in range(150):np.savetxt(self.root/'inputs/ego_pose'/f'{frame:03d}.txt',np.eye(4))
        factory=self.b.session.factory
        def make(spec):
            backend=factory(spec);body=Body();backend.trainer.models['SMPLNodes']=body
            backend.trainer.test_set_indices=[80,90,100,110,120]
            backend.env.models['SMPLNodes']={'instances_trans':body.instances_trans,'instances_quats':body.instances_quats}
            old_render=backend.render
            def render(pose,timestamp):
                body.cur_frame=int(getattr(backend,'_actor_source_frame',round(timestamp*10)));cam=SimpleNamespace(camtoworlds=torch.tensor(pose,dtype=torch.float32))
                self.last=body.get_gaussians(cam);result=old_render(pose,timestamp)
                # Raster double: encode actual target Gaussian movement as pixels.
                value=int(float(self.last['_means'][:3,1].sum())*15)%200+20
                for c in result['cameras'].values():c['rgb'][:]=value
                return result
            backend.render=render;return backend
        self.b.session.factory=make;self.f.load();self.manager=self.b.session.editor.actor_paths;self.body=self.manager.model
    def path(self,offset=2.):
        return dict(schema='actor_trajectory_v1',coordinate_frame='start_ego_ground',outside_interval='hold',waypoints=[dict(timestamp_s=8,offset_m=[0,0,0]),dict(timestamp_s=10,offset_m=[0,offset,0],yaw_delta_degrees=60)])
    def command(self,trajectory=None,operation='set_actor_trajectory'):
        value=dict(operation=operation,object_id='native:SMPLNodes:0')
        if operation=='set_actor_trajectory':value['trajectory']=trajectory or self.path()
        return value
    def edit(self,command=None):
        response=self.f.send('edit',self.f.args(command or self.command()));self.assertEqual(response['status'],'ok',response);return response
    def render(self,f):
        result=self.f.send('preview',dict(source_frame=f,expected_generation=self.b.session.generation,expected_revision=self.b.session.revision))
        self.assertEqual(result['status'],'ok',result);return result['result']
    def test_actual_gaussian_transform_preserves_articulation_and_other_body(self):
        self.render(90);source={k:v.clone() for k,v in self.last.items()};finger=self.manager.animation_fingerprint()
        identity=self.body.instances_trans;self.edit();packet=self.render(90)
        new=self.last;self.assertTrue(torch.equal(source['_means'][3:],new['_means'][3:]))
        for k in ('_opacities','_scales'):self.assertTrue(torch.equal(source[k],new[k]))
        self.assertTrue(torch.allclose(torch.cdist(source['_means'][:3],source['_means'][:3]),torch.cdist(new['_means'][:3],new['_means'][:3]),atol=1e-6))
        self.assertEqual(finger,self.manager.animation_fingerprint());self.assertIs(identity,self.body.instances_trans)
        evidence=packet['actor_trajectories']['render_evidence'][0]
        self.assertLess(evidence['rigid_shape_error_m'],1e-5);self.assertLess(evidence['proxy_error_m'],1e-5)
        self.assertEqual(evidence['gaussian_count'],3)
    def test_heldout_initial_root_preserves_exact_source_image(self):
        before=self.render(80);self.edit();after=self.render(80)
        self.assertEqual(before['cameras']['CAM_FRONT']['rgb_sha256'],after['cameras']['CAM_FRONT']['rgb_sha256'])
        self.assertAlmostEqual(float(self.body.instances_trans[80,0,0]),.8,places=6)
    def test_sh_uses_moved_view_and_inverse_edit_rotation(self):
        self.edit();self.render(90)
        from scipy.spatial.transform import Rotation
        r=torch.tensor(Rotation.from_rotvec([0,0,np.pi/6]).as_matrix(),dtype=torch.float32)
        pose=np.eye(4);camera=torch.tensor(pose[:3,3],dtype=torch.float32)
        # Recorded camera is identity in this fixture.
        dirs=torch.nn.functional.normalize(self.last['_means'][:3]-camera,dim=-1)@r
        coeff=torch.cat([self.body._features_dc[:3,None],self.body._features_rest[:3]],1)
        expected=(spherical_harmonics(1,dirs,coeff)+.5).clamp(0,1)
        self.assertTrue(torch.allclose(expected,self.last['_rgbs'][:3],atol=1e-6))
    def test_replacement_restore_reset_and_replay(self):
        original=self.body.instances_trans.clone();quats=self.body.instances_quats.clone()
        req=self.f.req('edit',self.f.args(self.command()),'once');self.assertEqual(self.b.handle(req)['status'],'ok');self.assertTrue(self.b.handle(req)['replayed'])
        self.assertEqual(self.b.session.revision,1);self.edit(self.command(self.path(1)))
        self.assertAlmostEqual(float(self.body.instances_trans[100,0,1]),1)
        self.edit(self.command(operation='restore_actor_trajectory'));self.assertTrue(torch.equal(original,self.body.instances_trans))
        self.edit();old=self.manager
        result=self.f.send('reset',dict(expected_generation=self.b.session.generation,expected_revision=4));self.assertEqual(result['status'],'ok',result)
        self.assertTrue(all(result['result']['reset_audit'].values()));self.assertTrue(old.closed)
        self.assertTrue(torch.equal(original,self.body.instances_trans));self.assertTrue(torch.equal(quats,self.body.instances_quats));self.render(80)
    def test_prior_static_edit_is_restored_by_path_then_full_reset(self):
        original=self.body.instances_trans.clone();self.edit(dict(operation='translate',object_id='native:SMPLNodes:0',offset_world_m=[.5,0,0]))
        static=self.body.instances_trans.clone();self.edit()
        response=self.f.send('edit',self.f.args(dict(operation='translate',object_id='native:SMPLNodes:0',offset_world_m=[.5,0,0])))
        self.assertEqual(response['error']['code'],'edit_conflict')
        self.edit(self.command(operation='restore_actor_trajectory'));self.assertTrue(torch.equal(static,self.body.instances_trans))
        self.b.session.reset();self.assertTrue(torch.equal(original,self.body.instances_trans))
    def test_invalid_paths_reject_without_mutation(self):
        original=self.body.instances_trans.clone()
        values=[]
        for field,value in [('timestamp_s',8),('timestamp_s',10.03),('offset_m',[0,8,0]),('yaw_delta_degrees',181)]:
            p=self.path();p['waypoints'][1][field]=value;values.append(p)
        self.body.instances_fv[85,0]=False;values.append(self.path())
        for p in values:
            response=self.f.send('edit',self.f.args(self.command(p)));self.assertEqual(response['status'],'error',response)
            self.assertEqual(self.b.session.revision,0);self.assertTrue(torch.equal(original,self.body.instances_trans))
    def test_original_body_failure_restores_planned_proxy(self):
        self.edit();planned=self.body.instances_trans.clone();self.body.fail=True
        self.assertEqual(self.f.send('preview',self.f.observation())['status'],'error')
        self.assertTrue(torch.equal(planned,self.body.instances_trans));self.assertIsNone(self.b.session.backend._last)
    def test_counterfactual_failure_restores_root_and_edit_phase(self):
        self.edit();planned=self.body.instances_trans.clone()
        with self.assertRaises(RuntimeError):
            with self.manager.original_paths():raise RuntimeError('interrupted baseline')
        self.assertFalse(self.manager.suspended);self.assertTrue(torch.equal(planned,self.body.instances_trans))
    def test_preview_video_and_runner_use_saved_actual_packets(self):
        self.edit();args=dict(expected_generation=self.b.session.generation,expected_revision=1,object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80)
        job=dict(schema='agent_job_v1',steps=[dict(request=self.f.req('preview_actor_trajectory',args))],verification=dict(minimum_coverage=dict(actor_trajectory_previews=1)))
        result=run_job(job,self.b,self.root/'job');self.assertTrue(result['passed']);self.assertEqual(result['rendered_packets'],6)
        self.assertFalse(result['gpu_render_executed']);self.assertFalse(self.b.session.editor.started)
        index=_hugsim_expand(json.loads((self.root/'job/actor_trajectory_index.json').read_text()));self.assertTrue(all(index[0]['audit'].values()));self.assertTrue(__import__('pathlib').Path(index[0]['closeup_video']).is_file())
        self.edit(self.command(operation='restore_actor_trajectory'))
    def test_gravity_basis_and_outside_interval_policy(self):
        from scipy.spatial.transform import Rotation
        tilted=np.eye(4);tilted[:3,:3]=Rotation.from_euler('xyz',[12,-7,3],degrees=True).as_matrix()
        for frame in (0,80):np.savetxt(self.root/'inputs/ego_pose'/f'{frame:03d}.txt',tilted)
        basis=ground_basis(self.b.session.spec,80);up=tilted[:3,:3].T@np.array([0,0,1.])
        self.assertTrue(np.allclose(basis[:,2],up));self.assertTrue(np.allclose(basis.T@basis,np.eye(3)))
        p=self.path();p['outside_interval']='original'
        plan=compile_path(p,self.b.session.spec,self.body.instances_trans[:,0].numpy(),np.ones(150,dtype=bool),[80])
        self.assertIsNone(sample_path(plan,7.9));self.assertIsNone(sample_path(plan,10.1))
        self.assertAlmostEqual(float((sample_path(plan,10)[0]-sample_path(plan,8)[0])@up),0,places=6)
    def test_failed_history_clear_rolls_back_edit_atomically(self):
        original=self.body.instances_trans.clone()
        with patch.object(self.b.session.backend,'clear_observation_history',side_effect=RuntimeError('clear failed')):
            response=self.f.send('edit',self.f.args(self.command()))
        self.assertEqual(response['status'],'error');self.assertFalse(self.manager.programs)
        self.assertEqual(self.b.session.revision,0);self.assertTrue(torch.equal(original,self.body.instances_trans))
    def test_preview_failure_leaves_incomplete_manifest(self):
        self.edit();self.body.fail=True
        response=self.f.send('preview_actor_trajectory',dict(expected_generation=self.b.session.generation,expected_revision=1,object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80),'failed_preview')
        self.assertEqual(response['status'],'error');manifest=_hugsim_expand(json.loads((self.b.output/'actor_trajectories/failed_preview/packets.json').read_text()))
        self.assertFalse(manifest['completed']);self.assertFalse(self.manager.suspended)

if __name__=='__main__':unittest.main()
