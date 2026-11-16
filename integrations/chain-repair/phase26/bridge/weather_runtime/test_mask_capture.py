import types
import weakref
import unittest
import torch
from mask_backend import install_mask_capture, MaskFrameworkBackend

class GS:
    def __init__(self,colors,opacity):self._rgbs=colors;self.opacity=opacity
    @property
    def rgbs(self):return self._rgbs

class MaskTests(unittest.TestCase):
    def make(self,labels,opacity,managed_label=None,protect_signals=False):
        t=types.SimpleNamespace(pts_labels=torch.tensor(labels),gaussian_classes={
            'Background':0,'RoadNodes':1,'TrafficLightNodes':2,'RigidNodes':3,'SMPLNodes':4,'DeformableNodes':5},info=None)
        if managed_label is not None:t.managed_dynamic_label=managed_label
        def raster(gs,cam,**kw):
            trans=torch.tensor(1.);rgb=torch.zeros(3)
            for color,alpha in zip(gs.rgbs,gs.opacity):rgb+=trans*alpha*color;trans=trans*(1-alpha)
            t.info='primary' if gs is original else 'semantic'
            return {'rgb_gaussians':rgb.reshape(1,1,3)},None
        original=GS(torch.ones(len(labels),3)*.3,torch.tensor(opacity))
        t.render_gaussians=raster;captured=install_mask_capture(t,protect_signals=protect_signals)
        result=t.render_gaussians(gs=original,cam=None)[0]['rgb_gaussians']
        return original,t,captured,result
    def test_static_occluder_preserved(self):
        gs,t,c,result=self.make([0,3],[1.,1.]);self.assertEqual(float(c[0][0,0]),0.)
        self.assertAlmostEqual(float(result[0,0,0]),.3,places=6)
    def test_roads_and_lights_are_not_dynamic(self):
        for label in (0,1,2):
            _,_,c,_=self.make([label],[1.]);self.assertEqual(float(c[0][0,0]),0.)
    def test_unified_weather_preserves_native_and_calibrated_signal_label(self):
        _,_,coverage,_=self.make([2],[1.],protect_signals=True)
        self.assertAlmostEqual(float(coverage[0][0,0]),1.)
        _,_,occluded,_=self.make([0,2],[1.,1.],protect_signals=True)
        self.assertEqual(float(occluded[0][0,0]),0.)
    def test_each_actor_class_is_dynamic(self):
        for label in (3,4,5):
            _,_,c,_=self.make([label],[.8]);self.assertAlmostEqual(float(c[0][0,0]),.8,places=6)
    def test_original_colors_and_render_info_unchanged(self):
        gs,t,_,_=self.make([3],[1.]);self.assertTrue(torch.equal(gs.rgbs,torch.ones(1,3)*.3))
        self.assertEqual(t.info,'primary')
    def test_managed_assets_are_dynamic_with_static_occlusion(self):
        _,_,c,_=self.make([6],[.8],managed_label=6)
        self.assertAlmostEqual(float(c[0][0,0]),.8,places=6)
        _,_,c,_=self.make([0,6],[1.,1.],managed_label=6)
        self.assertEqual(float(c[0][0,0]),0.)

    def test_close_releases_trainer_bound_methods_before_scene_reload(self):
        class Trainer:
            def render_gaussians(self, *args, **kwargs): pass
        trainer = Trainer()
        ref = weakref.ref(trainer)
        backend = MaskFrameworkBackend.__new__(MaskFrameworkBackend)
        backend.trainer = trainer
        backend._plain_raster = trainer.render_gaussians
        backend._coverage = [torch.zeros(3)]
        backend.env = types.SimpleNamespace(models={'RigidNodes': {'tensor': torch.zeros(3)}})
        del trainer
        backend.close()
        self.assertIsNone(ref())
        self.assertIsNone(backend._plain_raster)
        self.assertIsNone(backend.env)
        self.assertFalse(backend._coverage)
        backend.close()  # close is also safe after partial/previous cleanup

if __name__=='__main__':unittest.main()
