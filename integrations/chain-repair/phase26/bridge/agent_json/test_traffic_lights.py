
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

import copy,json
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_agent_bridge as fixtures
import agent_bootstrap as bootstrap
from scene_edit_session import saf
from agent_jobs import run_job
from traffic_lights import COLORS,state_at,validate_signal
from edit_api import read_command


class LightModel:
    def __init__(self):
        self.point_ids=torch.tensor([[0],[0],[0],[1],[2],[7],[7],[7]])
        rgb=torch.tensor([[1.,.1,0],[0.,1.,.1],[.2,.2,.2],[.9,.1,0],[.2,.2,.2],[0.,.9,.7],[0.,.8,.7],[0.,.9,.8]])
        self._features_dc=(rgb-.5)/.28209479177387814
        self._features_rest=torch.zeros(8,15,3);self._means=torch.arange(24,dtype=torch.float32).reshape(8,3)/100+torch.tensor([0.,0.,5.])
        self._scales=torch.full((8,3),.02);self._quats=torch.zeros(8,4);self._quats[:,0]=1
        self._opacities=torch.ones(8,1);self.instances_trans=torch.zeros(150,8,3)
        self.instances_quats=torch.zeros(150,8,4);self.instances_quats[...,0]=1
        self.instances_fv=torch.ones(150,8,dtype=torch.bool);self.cur_frame=0
    def get_gaussians(self):
        self.filter_mask=torch.ones(8,dtype=torch.bool)
        # Final learned color differs from canonical DC: override must run after it.
        rgb=torch.full((8,3),.2);rgb[[0,1,3,5,6,7]]=torch.tensor([.1,.7,.3])
        valid=self.instances_fv[self.cur_frame,self.point_ids.flatten()]
        return dict(_means=self._means,_scales=self._scales,_quats=self._quats,
            _opacities=self._opacities*valid[:,None],_rgbs=rgb)


class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.BridgeTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.bridge=self.f.bridge;self.root=self.f.root
        reader=saf.SceneSpec.read.side_effect
        def spec(path):
            value=reader(path);value.frames=150;return value
        saf.SceneSpec.read.side_effect=spec
        for i in [120,121,122]:np.savetxt(self.root/'inputs/ego_pose'/f'{i:03d}.txt',np.eye(4))
        factory=self.bridge.session.factory
        def make(value):
            backend=factory(value);model=LightModel();backend.trainer.models['TrafficLightNodes']=model
            render=backend.render
            def raster(pose,time):
                model.cur_frame=int(round(time*10));raw=render(pose,time);gs=model.get_gaussians()
                for cam in raw['cameras'].values():
                    cam['rgb']=cam['rgb'].copy();cam['rgb'][1,2]=torch.clamp(gs['_rgbs'][5]*255,0,255).numpy().astype(np.uint8)
                return raw
            backend.render=raster;return backend
        self.bridge.session.factory=make;self.f.load()
        self.manager=self.bridge.session.editor.traffic;self.model=self.manager.model
    def command(self,op='set_signal',index=0,**values):return dict(operation=op,object_id=f'native:TrafficLightNodes:{index}',**values)
    def edit(self,command):return self.f.send('edit',self.f.args(command))
    def test_discovery_distinguishes_absent_geometry_and_housing_only(self):
        result=self.f.send('list_traffic_lights');self.assertEqual(result['status'],'ok')
        rows={r['model_index']:r for r in result['result']['traffic_lights']}
        self.assertEqual(rows[0]['emitter_candidate_points'],2);self.assertEqual(rows[2]['operations'],[])
        self.assertEqual(rows[3]['point_count'],0)
        self.f.error(self.edit(self.command(index=3,state='red')),'unsupported_operation')
        self.f.error(self.edit(self.command(index=2,state='red')),'unsupported_operation')
        self.assertEqual(self.bridge.session.revision,0)
    def test_schema_rejects_geometry_ops_unknown_states_and_overlapping_programs(self):
        for cmd in [self.command(state='blue'),self.command('translate',offset_world_m=[1,0,0]),
                    self.command('set_signal_program',outside_state='original',phases=[dict(start_s=2,end_s=1,state='red')]),
                    self.command('set_signal_program',outside_state='original',phases=[dict(start_s=1,end_s=2,state='red'),dict(start_s=1.9,end_s=3,state='green')])]:
            with self.assertRaises(ValueError):read_command(dict(cmd,schema='scene_edit_v1',mode='diagnostic_preview'))
        self.f.error(self.edit(self.command('set_signal_program',outside_state='off',phases=[dict(start_s=14,end_s=16,state='red')])),'invalid_signal')
        self.assertEqual(self.bridge.session.revision,0)
    def test_color_override_is_after_network_and_changes_only_emitter_rgb(self):
        self.manager.timestamp=8.;before=self.model.get_gaussians();weights=self.model._features_dc.clone()
        response=self.edit(self.command(state='red'));self.assertEqual(response['status'],'ok',response)
        self.manager.timestamp=8.;after=self.model.get_gaussians()
        np.testing.assert_allclose(after['_rgbs'][:2].numpy(),[COLORS['red']]*2,atol=1e-7)
        self.assertTrue(torch.equal(before['_rgbs'][2:],after['_rgbs'][2:]))
        for key in ['_means','_scales','_quats','_opacities']:self.assertIs(before[key],after[key]) if key!='_opacities' else self.assertTrue(torch.equal(before[key],after[key]))
        self.assertTrue(torch.equal(self.model._features_dc,weights))
    def test_off_preserves_opacity_and_does_not_remove_light_geometry(self):
        self.manager.timestamp=8.;before=self.model.get_gaussians();self.edit(self.command(state='off'));self.manager.timestamp=8.
        after=self.model.get_gaussians();self.assertTrue(torch.equal(before['_opacities'],after['_opacities']))
        np.testing.assert_allclose(after['_rgbs'][0].numpy(),COLORS['off'],atol=1e-8)
    def test_half_open_program_boundaries_and_gaps_are_deterministic(self):
        p=dict(operation='set_signal_program',outside_state='original',phases=[dict(start_s=8,end_s=8.1,state='red'),dict(start_s=8.1,end_s=8.2,state='green')])
        validate_signal(p)
        self.assertEqual([state_at(p,t) for t in [7.9,8,np.nextafter(8.1,0),8.1,8.2]],['original','red','red','green','original'])
    def test_preview_commits_packet_state_but_leaves_editing_open(self):
        self.edit(self.command(index=7,state='red'));response=self.f.send('preview',dict(source_frame=120,expected_generation=self.bridge.session.generation,expected_revision=1))
        self.assertEqual(response['status'],'ok',response);signal=response['result']['traffic_signals']
        self.assertTrue(response['result']['edit_trust']['edited']);self.assertEqual(signal['programs'][0]['state'],'red');self.assertEqual(signal['last_render_receipts'][0]['valid_gaussian_points'],3)
        self.assertFalse(self.bridge.session.editor.started);self.assertIsNone(self.bridge.session.backend._last)
        self.assertEqual(self.edit(self.command(index=7,state='green'))['status'],'ok')
    def test_invalid_native_frame_is_not_claimed_visible(self):
        self.model.instances_fv[80,0]=False;self.edit(self.command(state='green'))
        response=self.f.send('preview',self.f.observation());self.assertEqual(response['status'],'ok')
        receipt=response['result']['traffic_signals']['last_render_receipts'][0]
        self.assertEqual(receipt['valid_gaussian_points'],0);self.assertFalse(receipt['camera_visibility_verified'])
    def test_restore_reset_and_request_replay_do_not_mutate_weights_or_stack_hooks(self):
        weights=self.model._features_dc.clone();before=self.f.send('preview',self.f.observation())
        req=self.f.req('edit',self.f.args(self.command(state='yellow')),'signal_once')
        self.assertEqual(self.bridge.handle(req)['status'],'ok');self.assertTrue(self.bridge.handle(req)['replayed'])
        self.assertEqual(self.bridge.session.revision,1)
        self.assertEqual(self.edit(self.command('restore_signal'))['status'],'ok')
        self.assertFalse(self.manager.programs)
        self.edit(self.command(state='red'));old=self.manager
        reset=self.f.send('reset',dict(expected_generation=self.bridge.session.generation,expected_revision=3))
        self.assertEqual(reset['status'],'ok');self.assertTrue(reset['result']['reset_audit']['traffic_overrides_cleared'])
        self.assertTrue(old.closed);self.assertIsNot(old,self.bridge.session.editor.traffic)
        self.assertTrue(torch.equal(weights,self.model._features_dc))
        after=self.f.send('preview',self.f.observation())
        self.assertEqual(before['result']['cameras']['CAM_FRONT']['rgb_sha256'],after['result']['cameras']['CAM_FRONT']['rgb_sha256'])
        self.assertTrue(self.bridge.handle(req)['replayed']);self.assertFalse(self.bridge.session.editor.traffic.programs)
    def test_signal_edit_rejected_after_formal_observation(self):
        self.f.send('observe',self.f.observation())
        self.f.error(self.edit(self.command(state='green')),'edit_requires_pre_run')
        self.assertFalse(self.manager.programs)
    def test_public_traffic_job_renders_all_states_and_resets(self):
        job=_hugsim_expand(json.loads((bootstrap.HERE/'examples/traffic_lights.json').read_text()))
        job['steps'][0]['request']['args'].update(expected_generation=self.bridge.session.generation,expected_revision=0)
        result=run_job(job,self.bridge,self.root/'traffic_job')
        self.assertTrue(result['passed']);self.assertEqual(result['completed_steps'],21)
        self.assertEqual(result['actual_renders'],10);self.assertEqual(result['coverage']['edits'],6)
        self.assertEqual(result['coverage']['resets'],1);self.assertFalse(result['closed_loop_driving'])
        index=_hugsim_expand(json.loads((self.root/'traffic_job/preview_index.json').read_text()))
        states={r['request_id']:r.get('traffic_signals',{}).get('programs',[]) for r in index}
        self.assertEqual(states['phase_red'][0]['state'],'red');self.assertEqual(states['phase_yellow'][0]['state'],'yellow')
        self.assertEqual(states['phase_green'][0]['state'],'green');self.assertEqual(states['reset_view'],[])
        from PIL import Image
        for state in ['red','yellow','green']:
            packet=next(r for r in index if r['request_id']=='phase_'+state)
            rgb=np.array(Image.open(packet['cameras']['CAM_FRONT']['rgb']))
            np.testing.assert_array_equal(rgb[1,2],(np.array(COLORS[state])*255).astype(np.uint8))
        self.assertTrue(result['checks']['signal_color_probe_passed'])
        self.assertTrue((self.root/'traffic_job/traffic_lights/signal_color_comparison.mp4').is_file())
    def test_clear_failure_rolls_back_program_and_revision(self):
        backend=self.bridge.session.backend
        def failed():raise RuntimeError('explicit clear failure')
        backend.clear_observation_history=failed
        result=self.edit(self.command(state='red'));self.f.error(result,'backend_error')
        self.assertFalse(self.manager.programs);self.assertEqual(self.bridge.session.revision,0)
    def test_output_alignment_failure_refuses_unproven_point_order(self):
        self.edit(self.command(state='red'));self.manager.timestamp=8.;g=self.manager.original_get();g['_rgbs']=g['_rgbs'][:-1]
        with self.assertRaises(RuntimeError):self.manager.override(g)

if __name__=='__main__':unittest.main()
