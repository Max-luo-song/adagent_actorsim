"""Motion continuity and actual selected-body interpolation with a CPU LBS double."""

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

import copy,json,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import test_actor_trajectories as original
from actor_motion import qmix,filter_quaternions,motion_sample
from actor_trajectories import compile_path,sample_path
from agent_jobs import run_job
spherical_harmonics=original.spherical_harmonics


class SmoothBody(original.Body):
    def __init__(self):
        super().__init__();self.in_test_set=False;self.fail_conditioned=False
        self.instances_trans[:,:,0]=torch.arange(150)[:,None]*.12
        f=np.arange(150);rotation=Rotation.from_rotvec(np.column_stack([np.zeros(150),np.zeros(150),np.deg2rad(2*(-1.)**f)])).as_quat()[:,[3,0,1,2]]
        self.instances_quats[:,:,0]=torch.tensor(rotation,dtype=torch.float32)[:,None]
        joint=Rotation.from_rotvec(np.column_stack([np.zeros(150),np.deg2rad(20*np.sin(f*.7)+4*(-1.)**f),np.zeros(150)])).as_quat()[:,[3,0,1,2]]
        self.smpl_qauts[:,:,0]=torch.tensor(joint,dtype=torch.float32)[:,None]
    def get_gaussians(self,cam):
        if self.fail or (self.fail_conditioned and not self.in_test_set):raise RuntimeError('body interpolation failed')
        f=self.cur_frame;ids=self.point_ids.flatten();trans=self.instances_trans[f].clone();root=self.instances_quats[f,:,0];joint=self.smpl_qauts[f,:,0]
        if self.in_test_set:
            trans=(self.instances_trans[f-1]+self.instances_trans[f+1])*.5
            root=torch.tensor(qmix(self.instances_quats[f-1,:,0].numpy(),self.instances_quats[f+1,:,0].numpy(),.5),dtype=torch.float32)
            joint=torch.tensor(qmix(self.smpl_qauts[f-1,:,0].numpy(),self.smpl_qauts[f+1,:,0].numpy(),.5),dtype=torch.float32)
        local=self._means.clone()
        for i in range(2):
            selection=ids==i;j=Rotation.from_quat(joint[i].numpy()[[1,2,3,0]]).as_matrix();r=Rotation.from_quat(root[i].numpy()[[1,2,3,0]]).as_matrix()
            articulated=local[selection].numpy().copy();articulated[1]=articulated[1]@j.T
            local[selection]=torch.tensor(articulated@r.T,dtype=torch.float32)
        means=local+trans[ids];coeff=torch.cat([self._features_dc[:,None],self._features_rest],1)
        rgb=(spherical_harmonics(1,torch.nn.functional.normalize(means-cam.camtoworlds[:3,3],dim=-1),coeff)+.5).clamp(0,1)
        self.filter_mask=torch.ones(6,dtype=torch.bool)
        return dict(_means=means,_quats=root[ids].clone(),_rgbs=rgb,_scales=torch.ones(6,3)*.1,_opacities=torch.ones(6,1))


class SmoothTests(unittest.TestCase):
    path=original.ActorTests.path
    command=original.ActorTests.command
    edit=original.ActorTests.edit
    render=original.ActorTests.render
    def setUp(self):
        with patch.object(original,'Body',SmoothBody):original.ActorTests.setUp(self)
        # Source trainer normally sets the test flag on every camera forward.
        render=self.manager.original_render
        def wrapped(pose,timestamp):
            self.body.in_test_set=int(getattr(self.b.session.backend,'_actor_source_frame',round(timestamp*10))) in self.manager.tests
            return render(pose,timestamp)
        self.manager.original_render=wrapped
    def smooth(self):
        p=self.path();p['waypoints'][1].pop('yaw_delta_degrees');p.update(motion_profile='smooth_v1',heading_mode='path_tangent');return p
    def observe_time(self,t):
        return self.f.send('preview',dict(expected_generation=self.b.session.generation,expected_revision=self.b.session.revision,timestamp_s=t,ego_to_world=np.eye(4).tolist()))
    def test_path_has_continuous_velocity_and_bounded_speed(self):
        p=self.smooth();p['waypoints'].insert(1,dict(timestamp_s=9,offset_m=[.15,1,0]))
        plan=compile_path(p,self.b.session.spec,self.body.instances_trans[:,0].numpy(),np.ones(150,dtype=bool),self.manager.tests)
        a=plan['position_spline'](9-1e-6,1);b=plan['position_spline'](9+1e-6,1)
        self.assertLess(np.linalg.norm(a-b),1e-4)
        self.assertTrue(np.allclose(sample_path(plan,8)[0],[9.6,0,0],atol=1e-5))
        self.assertTrue(np.allclose(sample_path(plan,10)[0],[9.6,2,0],atol=1e-5))
    def test_subframes_change_real_body_and_restore_source_parameters(self):
        self.edit(self.command(self.smooth()));finger=self.manager.animation_fingerprint();roots=self.body.instances_trans.clone();quats=self.body.instances_quats.clone()
        a=self.observe_time(8.0);ga=self.last['_means'].clone()
        b=self.observe_time(8+1/30);gb=self.last['_means'].clone()
        self.assertEqual(a['status'],'ok',a);self.assertEqual(b['status'],'ok',b)
        self.assertFalse(torch.equal(ga[:3],gb[:3]));self.assertTrue(torch.equal(ga[3:],gb[3:]))
        # Same source frame, different joint phase and shape, not just a root shift.
        self.assertNotAlmostEqual(float(torch.linalg.vector_norm(ga[1]-ga[2])),float(torch.linalg.vector_norm(gb[1]-gb[2])),places=5)
        evidence=b['result']['actor_trajectories']['render_evidence'][0]
        self.assertGreater(evidence['animation_frame'],80);self.assertLess(evidence['animation_frame'],81)
        self.assertTrue(evidence['gait_retimed']);self.assertLess(evidence['proxy_error_m'],1e-4)
        self.assertEqual(finger,self.manager.animation_fingerprint());self.assertTrue(torch.equal(roots,self.body.instances_trans));self.assertTrue(torch.equal(quats,self.body.instances_quats))
        self.assertEqual(b['result']['source_frame'],80);self.assertIsNone(self.manager.runtime_frame)
    def test_heading_stable_when_source_root_heading_jitters(self):
        self.edit(self.command(self.smooth()));head=[]
        for t in [8+i/30 for i in range(10)]:
            res=self.observe_time(t);self.assertEqual(res['status'],'ok',res);head.append(res['result']['actor_trajectories']['render_evidence'][0]['yaw_delta_degrees'])
        self.assertLess(np.ptp(head),1e-4)
    def test_gait_phase_tracks_path_distance_and_smooths_noise(self):
        self.edit(self.command(self.smooth()));entry=self.manager.programs[0]
        a=motion_sample(entry,8);b=motion_sample(entry,10)
        self.assertAlmostEqual(b['animation_frame']-a['animation_frame'],2/.12,places=3)
        source=self.body.smpl_qauts[:,0].numpy();filtered=entry['motion']['bones']
        # A local alternating perturbation has lower energy after filtering.
        self.assertLess(np.mean(np.diff(filtered[81:99],n=2,axis=0)**2),np.mean(np.diff(source[81:99],n=2,axis=0)**2))
    def test_parameter_restore_on_failed_conditioned_forward(self):
        self.edit(self.command(self.smooth()));joints=self.body.smpl_qauts.clone();roots=self.body.instances_trans.clone();quats=self.body.instances_quats.clone()
        self.body.fail_conditioned=True
        result=self.observe_time(8);self.assertEqual(result['status'],'error',result)
        self.assertTrue(torch.equal(joints,self.body.smpl_qauts));self.assertTrue(torch.equal(roots,self.body.instances_trans));self.assertTrue(torch.equal(quats,self.body.instances_quats))
        self.assertIsNone(self.manager.runtime_frame);self.assertIsNone(self.b.session.backend._last)
    def test_subframe_preserves_exact_scene_time_for_other_modules(self):
        self.edit(self.command(self.smooth()));observed=[];render=self.manager.original_render
        def spy(pose,timestamp):
            observed.append((timestamp,getattr(self.b.session.backend,'_actor_source_frame',None)))
            return render(pose,timestamp)
        self.manager.original_render=spy;timestamp=8+2/30
        result=self.observe_time(timestamp);self.assertEqual(result['status'],'ok',result)
        self.assertEqual(observed,[(timestamp,80)]);self.assertEqual(result['result']['source_frame'],80)
        self.assertFalse(hasattr(self.b.session.backend,'_actor_source_frame'))
    def test_incompatible_comparison_baseline_rejected_before_render(self):
        self.edit(self.command(self.smooth()));count=self.b.render_count
        result=self.f.send('preview_actor_trajectory',dict(expected_generation=self.b.session.generation,expected_revision=1,object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80,render_fps=30,comparison_baseline='scene552_pedestrian_v1'))
        self.assertEqual(result['error']['code'],'invalid_motion_baseline');self.assertEqual(self.b.render_count,count)
        self.assertFalse(self.b.session.editor.started);self.assertIn(0,self.manager.programs)
    def test_restore_and_reset_after_fractional_render(self):
        roots=self.body.instances_trans.clone();joints=self.body.smpl_qauts.clone();clear=self.manager.original_clear
        self.edit(self.command(self.smooth()));self.observe_time(8+1/30)
        self.edit(self.command(operation='restore_actor_trajectory'));self.assertTrue(torch.equal(roots,self.body.instances_trans))
        self.edit(self.command(self.smooth()));self.b.session.reset()
        self.assertTrue(torch.equal(roots,self.body.instances_trans));self.assertTrue(torch.equal(joints,self.body.smpl_qauts));self.assertTrue(self.manager.closed)
        self.assertEqual(self.b.session.editor.actor_paths.original_clear,clear)
    def test_quaternion_sign_flip_and_invalid_gap_do_not_create_spikes(self):
        q=np.tile([1.,0,0,0],(9,1));q[::2]*=-1
        out=filter_quaternions(q,np.ones(9,dtype=bool));self.assertTrue(np.allclose(abs(out[:,0]),1))
        self.assertTrue(np.allclose(abs(qmix(q[0],q[1],.5)),[1,0,0,0]))
        q[5:]=[0,0,0,1];valid=np.ones(9,dtype=bool);valid[4]=False
        out=filter_quaternions(q,valid);self.assertTrue(np.allclose(abs(out[3]),[1,0,0,0]))
    def test_missing_animation_or_excess_retime_rejected_before_mutation(self):
        roots=self.body.instances_trans.clone();self.body.instances_fv[85:,0]=False
        r=self.f.send('edit',self.f.args(self.command(self.smooth())));self.assertEqual(r['status'],'error')
        self.assertEqual(self.b.session.revision,0);self.assertTrue(torch.equal(roots,self.body.instances_trans))
    def test_smooth_preview_30fps_runs_full_evidence_and_video(self):
        self.edit()
        baseline=dict(completed=True,object_id='native:SMPLNodes:0',camera_source_frame=80,frames=[dict(new=self.render(f)) for f in range(80,83)])
        baseline_path=self.root/'baseline.json';baseline_path.write_text(json.dumps(baseline))
        self.edit(self.command(operation='restore_actor_trajectory'));self.edit(self.command(self.smooth()))
        args=dict(expected_generation=self.b.session.generation,expected_revision=3,object_id='native:SMPLNodes:0',start_frame=80,end_frame=82,camera_source_frame=80,render_fps=30,comparison_baseline='fixture')
        job=dict(schema='agent_job_v1',steps=[dict(request=self.f.req('preview_actor_trajectory',args))])
        import actor_motion_review
        with patch.dict(actor_motion_review.BASELINES,{'fixture':baseline_path}):result=run_job(job,self.b,self.root/'smooth_job')
        self.assertTrue(result['passed']);self.assertEqual(result['rendered_packets'],14)
        index=_hugsim_expand(json.loads((self.root/'smooth_job/actor_trajectory_index.json').read_text()));self.assertEqual(index[0]['render_fps'],30);self.assertTrue(index[0]['gait_retimed']);self.assertTrue(index[0]['motion_comparison']['passed']);self.assertTrue(result['checks']['actor_motion_comparison_verified'])

if __name__=='__main__':unittest.main()
