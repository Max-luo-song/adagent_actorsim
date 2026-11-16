"""Diagnostic layer isolation must retain the production render and geometry."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from run_render_quality import LayerCapture, native_render_center

class Gaussians(SimpleNamespace):
    @property
    def opacities(self):return self._opacities
    @property
    def rgbs(self):return self._rgbs

class LayerTests(unittest.TestCase):
    def setUp(self):
        model=SimpleNamespace(point_ids=torch.tensor([[0],[1],[1]]),filter_mask=torch.tensor([True,True,True]))
        self.gs=Gaussians(_opacities=torch.tensor([[.1],[.2],[.3],[.4],[.5]]),
            _rgbs=torch.arange(15).reshape(5,3).float()/15)
        self.calls=[]
        self.trainer=SimpleNamespace(models={'RigidNodes':model},pts_labels=torch.tensor([0,1,1,1,2]),
            gaussian_classes={'Background':0,'RigidNodes':1,'SMPLNodes':2},info='initial')
        def render(gs,cam,**kwargs):
            self.calls.append((gs.opacities.clone(),gs.rgbs.clone()))
            self.trainer.info=f'call{len(self.calls)}'
            opacity=gs.opacities.sum().reshape(1,1,1)
            rgb=(gs.rgbs*gs.opacities).sum(0).reshape(1,1,3)
            return dict(rgb_gaussians=rgb,opacity=opacity,depth=torch.ones(1,1,1)), 'production_fn'
        self.trainer.render_gaussians=render
        self.original=render
        self.capture=LayerCapture(SimpleNamespace(trainer=self.trainer));self.addCleanup(self.capture.close)

    def test_target_background_layers_partition_same_opacities_without_touching_output(self):
        before_opacity=self.gs.opacities.clone();before_rgb=self.gs.rgbs.clone()
        out,fn=self.trainer.render_gaussians(self.gs,object())
        self.assertEqual(fn,'production_fn');self.assertEqual(len(self.calls),4)
        torch.testing.assert_close(out['rgb_gaussians'],(before_rgb*before_opacity).sum(0).reshape(1,1,3))
        torch.testing.assert_close(self.calls[1][0]+self.calls[3][0],before_opacity)
        torch.testing.assert_close(self.calls[1][0],torch.tensor([[0.],[0.],[.3],[.4],[0.]]))
        self.assertTrue(torch.equal(self.calls[2][1],torch.full((5,3),.6)))
        self.assertTrue(torch.equal(self.gs.opacities,before_opacity));self.assertTrue(torch.equal(self.gs.rgbs,before_rgb))
        self.assertEqual(self.trainer.info,'call1')
        self.trainer.render_gaussians(self.gs,object());self.assertEqual(len(self.calls),5)
        self.capture.close();self.assertIs(self.trainer.render_gaussians,self.original)

    def test_class_layers_preserve_production_output_and_split_other_instances(self):
        self.capture.capture_groups=True
        out,fn=self.trainer.render_gaussians(self.gs,object())
        self.assertEqual(len(self.calls),7)
        row=self.capture.rows[0]
        self.assertIn('group_Background',row)
        self.assertIn('group_other_RigidNodes',row)
        np.testing.assert_allclose(row['group_other_RigidNodes_alpha'],.2)
        np.testing.assert_allclose(row['group_Background_alpha'],.1)
        self.assertEqual(self.trainer.info,'call1')
        self.assertEqual(fn,'production_fn')

    def test_orbit_uses_rendered_center_at_held_out_frames(self):
        t=torch.zeros(90,2,3);t[79,1]=torch.tensor([1.,2.,3.]);t[81,1]=torch.tensor([3.,4.,5.]);t[80,1]=99
        m=SimpleNamespace(instances_trans=t,instances_fv=torch.ones(90,2,dtype=torch.bool))
        trainer=SimpleNamespace(models={'RigidNodes':m},test_set_indices=[80])
        center,interpolated=native_render_center(trainer)
        np.testing.assert_array_equal(center,[2,3,4]);self.assertTrue(interpolated)
        m.instances_fv[79,1]=False
        center,interpolated=native_render_center(trainer)
        np.testing.assert_array_equal(center,[99,99,99]);self.assertFalse(interpolated)
        trainer.test_set_indices=[];m.instances_fv[79,1]=True
        center,interpolated=native_render_center(trainer)
        np.testing.assert_array_equal(center,[99,99,99]);self.assertFalse(interpolated)

    def test_managed_layer_uses_its_shared_collection_label(self):
        self.capture.target_kind='managed';self.trainer.managed_dynamic_label=3
        self.trainer.pts_labels=torch.tensor([0,1,1,3,3])
        self.trainer.render_gaussians(self.gs,object())
        torch.testing.assert_close(self.calls[1][0],torch.tensor([[0.],[0.],[0.],[.4],[.5]]))

    def test_identity_mismatch_stops_ablation_instead_of_mixing_wrong_instance(self):
        self.trainer.models['RigidNodes'].point_ids=torch.tensor([[1]])
        with self.assertRaisesRegex(RuntimeError,'IDs and collection differ'):
            self.trainer.render_gaussians(self.gs,object())

if __name__=='__main__':unittest.main()
