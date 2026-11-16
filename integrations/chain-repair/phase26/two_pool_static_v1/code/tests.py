import unittest,copy
import numpy as np
from shapely.geometry import box
from common import *
from ego_mask import rasterize,exclude,load_mask,REVISION
from prepare import episodes
from projection_domain import valid_ray

class StaticTests(unittest.TestCase):
 def test_missing_ego_mask_never_silently_empty(self):
  with self.assertRaises(ValueError):rasterize([], (10,8))
  with self.assertRaises(ValueError):load_mask({},0,(8,10))
 def test_explicit_empty_requires_authority(self):
  self.assertFalse(rasterize([], (10,8),empty_reviewed=True).any())
 def test_ego_road_and_nonroad_both_become_unknown(self):
  mask=rasterize([[(2,2),(5,2),(5,5),(2,5)]],(8,8))>0
  labels=np.array([0,2,0]);pixels=np.array([[3,3],[4,4],[0,0]]);valid=np.ones(3,bool);before=valid.copy();before_labels=labels.copy()
  np.testing.assert_array_equal(exclude(valid,pixels,mask),[False,False,True]);np.testing.assert_array_equal(valid,before);np.testing.assert_array_equal(labels,before_labels)
 def test_non_ego_obstacle_not_removed(self):
  mask=rasterize([[(2,2),(5,2),(5,5),(2,5)]],(8,8))>0
  np.testing.assert_array_equal(exclude(np.ones(2,bool),np.array([[1,1],[6,6]]),mask),[True,True])
 def test_bad_mask_geometry_rejected(self):
  with self.assertRaises(ValueError):rasterize([[(0,0),(10,0),(2,2)]],(8,8))
 def test_ring_cannot_pass_hole_or_boundary(self):
  body=box(-1.5,-.8,1.5,.8);ring=body.buffer(.5,quad_segs=32)
  self.assertTrue(box(-3,-3,3,3).covers(ring));self.assertFalse(box(-1.9,-2,2,2).covers(ring));self.assertFalse(box(-3,-3,3,3).difference(box(-.2,-.2,.2,.2)).covers(ring))
 def test_sustained_episodes_do_not_bridge_unknown_frames(self):
  self.assertEqual(episodes([False]+[True]*5+[False]+[True]*4),[(1,5)])
 def test_projection_still_rejects_disconnected_lens_branch(self):
  k=np.array([100,100,50,50,-1,0,0,0,0.]);self.assertFalse(valid_ray(np.array([[1.2,0,1.]]),k)[0]);self.assertTrue(valid_ray(np.array([[.1,0,1.]]),k)[0])
 def test_real_3_core_cannot_be_trimmed_for_six_meters(self):
  a=read(ASSETS/'3/necessary_admission.json');self.assertEqual(a['core_interval'][0],0);self.assertGreater(a['first_6m_frame'],0);self.assertFalse(a['core_preserving_candidates'])
 def test_real_23_never_reaches_six_meters(self):
  a=read(ASSETS/'23/necessary_admission.json');self.assertLess(a['total_travel_m'],6);self.assertIsNone(a['first_6m_frame'])
 def test_real_late_motion_cannot_leave_ten_seconds(self):
  for scene in (172,621):
   a=read(ASSETS/str(scene)/'necessary_admission.json');self.assertLess(a['first_6m_remaining_seconds'],10);self.assertFalse(a['window_qualified_frames'])
 def test_real_233_core_must_not_be_truncated(self):
  a=read(ASSETS/'233/necessary_admission.json');self.assertEqual(a['core_interval'][0],0);self.assertGreater(a['first_6m_frame'],0);self.assertFalse(a['core_preserving_candidates'])
 def test_actual_map_filters_ego_before_votes_without_filling(self):
  source=(LOCAL/'boundaries.py').read_text();self.assertIn('valid=exclude(valid,pixels,load_mask',source);self.assertIn("hole_filling_applied':False",source);self.assertNotIn('binary_closing',source)
 def test_weight_identity_and_no_phase14_loader(self):
  a=read(OUT/'weight_lineage_audit.json');self.assertTrue(a['matches_pinned_runtime_hash']);self.assertFalse(a['phase14_load_path_in_selected_boot_sources'])
if __name__=='__main__':unittest.main(verbosity=2)
