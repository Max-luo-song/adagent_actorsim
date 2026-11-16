from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_profiles import resolve_model_profile,apply_model_profile
class ModelProfilesTests(unittest.TestCase):
    def test_metadata_selects_one_shared_geometry_profile_without_threshold_changes(self):
        _,legacy=resolve_model_profile({})
        for scene in ['new-scene',172,552,'023','621']:
            name,profile=resolve_model_profile(dict(scene=scene,model_input_profile='calibration_geometry_v1'))
            self.assertEqual(name,'calibration_geometry_v1')
            self.assertEqual({k:v for k,v in profile.items() if legacy[k]!=v},dict(resize_mode='letterbox',camera_embedding_policy='geometry_only'))
    def test_profile_disables_prior_without_touching_embedding_weights(self):
        embedding=object();module=SimpleNamespace(use_cams_embeds=True,cams_embeds=embedding)
        core=SimpleNamespace(modules=lambda:[module]);_,profile=resolve_model_profile(dict(model_input_profile='calibration_geometry_v1'))
        self.assertEqual(apply_model_profile(core,profile),1)
        self.assertFalse(module.use_cams_embeds)
        self.assertIs(module.cams_embeds,embedding)
    def test_unknown_and_ambiguous_profiles_rejected(self):
        for seq in [dict(model_input_profile='scene_offset_172'),dict(model_input_profile='calibration_geometry_v1',diagnostic_variants=['reference'])]:
            with self.assertRaises(ValueError):resolve_model_profile(seq)
