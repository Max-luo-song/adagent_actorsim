import unittest,copy
import numpy as np
from shapely.geometry import box
from common import *
from scan import qualified,body_polygon
from ego_mask import load_mask,exclude
class Tests(unittest.TestCase):
 def test_causal_warmup_correction_real3(self):
  old=read(PREVIOUS/'assets/3/necessary_admission.json');new=read(ASSETS/'3/interaction_registration_v2.json');self.assertFalse(old['core_preserving_candidates']);self.assertEqual(new['primary_peak_frame'],32);self.assertIn(13,new['candidate_frames_before_map']);self.assertNotIn(33,new['candidate_frames_before_map'])
 def test_causal_nominal_real31_not_rejected_for_missing_aggression(self):
  old=read(PREVIOUS/'assets/31/necessary_admission.json');new=read(ASSETS/'31/interaction_registration_v2.json');self.assertIsNone(old['core_interval']);self.assertEqual(new['stratum'],'N');self.assertIn(10,new['candidate_frames_before_map'])
 def test_233_peak_is_not_moved_by_later_start(self):
  new=read(ASSETS/'233/interaction_registration_v2.json');self.assertEqual(new['primary_peak_frame'],50);self.assertIn(50,new['candidate_frames_before_map']);self.assertNotIn(51,new['candidate_frames_before_map'])
 def test_real552_nominal_six_meter_start(self):
  new=read(ASSETS/'552/interaction_registration_v2.json');self.assertIn(58,new['candidate_frames_before_map']);self.assertNotIn(57,new['candidate_frames_before_map'])
 def test_map_is_mandatory_even_if_motion_and_core_pass(self):
  r={'base_motion_window_pass':True,'interaction_retention_pass':True};self.assertFalse(qualified(r,False));self.assertTrue(qualified(r,True))
 def test_peak_and_window_cannot_be_cancelled_by_full_map(self):
  for motion,peak in ((False,True),(True,False)):
   self.assertFalse(qualified({'base_motion_window_pass':motion,'interaction_retention_pass':peak},True))
 def test_actual_source_masks_are_explicit_and_shape_bound(self):
  for scene in SCENES:
   src=read(ASSETS/str(scene)/'source_manifest.json');m=read(ASSETS/str(scene)/'ego_masks/manifest.json');verify(m['source_manifest'])
   for cam in range(5):
    wh=src['frames_data'][0]['image_size_wh'][str(cam)];mask=load_mask(m,cam,wh[::-1]);self.assertEqual(mask.shape,tuple(wh[::-1]));self.assertTrue(m['cameras'][str(cam)]['explicit_reviewed_empty'])
 def test_ego_exclusion_does_not_create_road_votes(self):
  mask=np.zeros((3,3),bool);mask[1,1]=True;valid=np.ones(2,bool);pixels=np.array([[1,1],[0,0]]);original=valid.copy();np.testing.assert_array_equal(exclude(valid,pixels,mask),[False,True]);np.testing.assert_array_equal(valid,original)
 def test_full_body_ring_clearance_must_preserve_holes(self):
  body=body_polygon(np.eye(4));poly=box(-3,-3,3,3).difference(box(-.1,-.1,.1,.1));self.assertFalse(poly.covers(body));self.assertLess(body.distance(box(-1.9,-2,2,2).boundary),.5)
 def test_rule_digest_frozen(self):self.assertEqual(sha(OUT/'preregistration.json'),(OUT/'preregistration.sha256').read_text().split()[0])
if __name__=='__main__':unittest.main(verbosity=2)
