import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import agent_bootstrap
from calibrated_signals import CalibratedSignals,build_geometry,load_profiles
from traffic_lights import COLORS
from edit_api import read_command
import test_traffic_lights as fixtures


def profile():
    return dict(schema='calibrated_signal_lenses_v1',profile_id='demo',green_center_source=[0.,0.,5.],
        up_source=[0.,1.,0.],normal_towards_viewer_source=[0.,0.,-1.],lens_radius_m=.14,lens_spacing_m=.31,
        surface_front_bias_m=.1,gaussian_spacing_m=.035,gaussian_sigma_m=.021,gaussian_opacity=.995)


class CalibratedTests(unittest.TestCase):
    def backend(self):
        fields=dict(_means=torch.zeros(2,3),_scales=torch.ones(2,3),_quats=torch.tensor([[1.,0,0,0]]*2),_opacities=torch.ones(2,1),_rgbs=torch.ones(2,3)*.2)
        self.gs=SimpleNamespace(**fields);self.freeze=torch.zeros(2,dtype=torch.bool)
        trainer=SimpleNamespace(gaussian_classes={'Background':0,'TrafficLightNodes':4},pts_labels=torch.tensor([0,4]),dynamic_pts_mask=torch.tensor([False,True]))
        def collect(*args,**kwargs):trainer.pts_labels=torch.tensor([0,4]);return self.gs,self.freeze
        trainer.collect_gaussians=collect;self.original=collect
        backend=SimpleNamespace(torch=torch,trainer=trainer)
        with patch('calibrated_signals.load_profiles',return_value={'demo':profile()}):self.layer=CalibratedSignals(backend,SimpleNamespace(),lambda:12.)
        self.addCleanup(self.layer.close);self.b=backend
        pose=torch.eye(4);self.cam=SimpleNamespace(camtoworlds=pose)
        return backend
    def test_geometry_has_three_separate_tight_lenses_and_valid_quaternions(self):
        g=build_geometry(profile());self.assertEqual(set(g['slots']),{0,1,2})
        np.testing.assert_allclose(g['centers'][0]-g['centers'][1],[0,.31,0]);np.testing.assert_allclose(g['centers'][1]-g['centers'][2],[0,.31,0])
        self.assertTrue(np.all(g['opacities']>.99));self.assertLess(g['scales'].max(),.03)
        np.testing.assert_allclose(np.linalg.norm(g['quats'],axis=1),1)
    def test_shared_raster_append_has_dark_inactive_slots_and_keeps_original_fields(self):
        b=self.backend();self.layer.programs['demo']=dict(operation='set_signal',state='red')
        merged,freeze=b.trainer.collect_gaussians(cam=self.cam);g=self.layer.geometry['demo'];n=len(g['means'])
        self.assertEqual(len(merged._means),n+2);self.assertTrue(torch.equal(merged._means[:2],self.gs._means))
        self.assertEqual(len(freeze),n+2);self.assertTrue(torch.equal(b.trainer.pts_labels[-n:],torch.full((n,),4)))
        colors=merged._rgbs[2:].numpy();np.testing.assert_allclose(colors[g['slots']==0],[COLORS['red']]*sum(g['slots']==0),atol=1e-7)
        self.assertTrue(np.all(colors[g['slots']!=0]<.01));self.assertTrue(torch.equal(self.gs._rgbs,torch.ones(2,3)*.2))
    def test_color_changes_do_not_move_lenses_or_change_alpha(self):
        b=self.backend();self.layer.programs['demo']=dict(operation='set_signal',state='red');red,_=b.trainer.collect_gaussians(cam=self.cam)
        self.layer.programs['demo']['state']='green';green,_=b.trainer.collect_gaussians(cam=self.cam)
        self.assertTrue(torch.equal(red._means,green._means));self.assertTrue(torch.equal(red._opacities,green._opacities))
        self.assertGreater(float((red._rgbs-green._rgbs).abs().max()),.9)
    def test_original_and_backface_do_not_append_and_close_restores_collector(self):
        b=self.backend();self.layer.programs['demo']=dict(operation='set_signal',state='original');gs,_=b.trainer.collect_gaussians(cam=self.cam);self.assertIs(gs,self.gs)
        self.layer.programs['demo']['state']='green';self.cam.camtoworlds[2,3]=10
        gs,_=b.trainer.collect_gaussians(cam=self.cam);self.assertIs(gs,self.gs);self.assertFalse(self.layer.receipts[0]['front_facing'])
        self.layer.close();self.assertIs(b.trainer.collect_gaussians,self.original)
    def test_invalid_geometry_and_cross_scene_profile_are_rejected(self):
        bad=profile();bad['lens_spacing_m']=.1
        with self.assertRaises(ValueError):build_geometry(bad)
        self.assertFalse(load_profiles(SimpleNamespace(source={'scene':551,'checkpoint':'/tmp/unrelated.pth'})))
        self.assertFalse(load_profiles(SimpleNamespace(source={'scene':552,'checkpoint':'/tmp/unrelated.pth'})))
    def test_visual_gate_rejects_halo_and_simultaneously_lit_green(self):
        from traffic_presentation import lens_color_checks
        colors={'red':[245.,5,5],'yellow':[245.,200,5],'green':[5.,245,10]}
        values={state+'_view':{slot:(colors[slot] if state==slot else [15.,15,15]) for slot in colors} for state in ['red','yellow','green','off']}
        self.assertTrue(all(lens_color_checks(values).values()))
        bad=copy.deepcopy(values);bad['red_view']['red']=[171.,150,150]
        self.assertFalse(lens_color_checks(bad)['calibrated_active_lens_is_clear_color'])
        bad=copy.deepcopy(values);bad['red_view']['green']=[5.,245,10]
        self.assertFalse(lens_color_checks(bad)['calibrated_inactive_lenses_are_dark'])

    def test_registered_signal_uses_same_json_revision_reset_and_edited_flag(self):
        f=fixtures.TrafficTests();f.setUp();self.addCleanup(f.doCleanups)
        manager=f.bridge.session.editor.traffic;manager.calibrated.profiles['demo']=profile();manager.calibrated.geometry['demo']=build_geometry(profile())
        cmd=dict(operation='set_signal',object_id='signal:demo',state='red')
        parsed=read_command(dict(cmd,schema='scene_edit_v1',mode='diagnostic_preview'));self.assertEqual(parsed['state'],'red')
        result=f.edit(cmd);self.assertEqual(result['status'],'ok',result)
        self.assertTrue(result['result']['receipt']['evidence']['geometry_changed']);self.assertTrue(manager.has_programs)
        response=f.f.send('preview',f.f.observation());self.assertTrue(response['result']['edit_trust']['edited'])
        reset=f.f.send('reset',dict(expected_generation=f.bridge.session.generation,expected_revision=1))
        self.assertTrue(reset['result']['reset_audit']['traffic_overrides_cleared']);self.assertFalse(f.bridge.session.editor.traffic.has_programs)

if __name__=='__main__':unittest.main()
