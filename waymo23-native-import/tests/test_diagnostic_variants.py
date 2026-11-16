from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostic_variants import requested_variants, configure_variant, restore_variant

class DiagnosticVariantsTests(unittest.TestCase):
    def test_production_cannot_opt_into_experiments(self):
        with self.assertRaises(ValueError):
            requested_variants({'diagnostic_variants':['fresh_frame']}, False)
        self.assertIsNone(requested_variants({},False))

    def test_unknown_duplicate_or_unbounded_profiles_fail(self):
        for values in ([],['foo'],['reference','reference'],'reference',[{},'reference']):
            with self.assertRaises(ValueError):
                requested_variants({'diagnostic_variants':values},True)

    def test_switches_restore_without_changing_embedding_weights(self):
        module=SimpleNamespace(use_cams_embeds=True,cams_embeds=object())
        core=SimpleNamespace(modules=lambda:[module])
        weights=module.cams_embeds
        switches=configure_variant(core,'no_camera_embedding')
        try:
            self.assertFalse(module.use_cams_embeds)
            self.assertIs(module.cams_embeds,weights)
        finally:
            restore_variant(switches)
        self.assertTrue(module.use_cams_embeds)
        self.assertEqual(requested_variants({'diagnostic_variants':['reference','fresh_frame']},True),['reference','fresh_frame'])

    def test_missing_feature_switch_fails_explicitly(self):
        with self.assertRaises(ValueError):
            configure_variant(SimpleNamespace(modules=lambda:[]),'no_camera_embedding')

    def test_pillar_probe_moves_only_projection_height_and_restores(self):
        import torch
        class Encoder:
            num_points_in_pillar=4
            pc_range=[-51.2,-51.2,-5.,51.2,51.2,3.]
            def point_sampling(self,reference_points,pc_range,img_metas):return reference_points
        encoder=Encoder();module=SimpleNamespace(use_cams_embeds=True)
        core=SimpleNamespace(modules=lambda:[module,encoder]);original=encoder.point_sampling
        points=torch.tensor([[[.3,.4,.5]]]);before=points.clone()
        state=configure_variant(core,'letterbox_no_camera_embedding_pillar_down')
        actual=encoder.point_sampling(points,encoder.pc_range,[])
        torch.testing.assert_close(actual[...,:2],before[...,:2])
        torch.testing.assert_close(actual[...,2],before[...,2]-.5/8.)
        torch.testing.assert_close(points,before)
        restore_variant(state)
        self.assertEqual(encoder.point_sampling,original)
        self.assertTrue(module.use_cams_embeds)
