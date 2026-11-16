
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

import sys,unittest
pass  # Path is provided by the portable boundary above
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from start_diagnostics import classify,PoseBuckets,cells_for_regions,footprint
class StartAuditTests(unittest.TestCase):
 def test_four_states_separate_nonroad_from_road_only_insufficiency(self):
  np.testing.assert_array_equal(classify([0,1,0,1,2,4],[0,1,1,3,3,5]),[0,1,2,2,2,3])
 def test_vote_corruption_is_rejected(self):
  with self.assertRaises(ValueError):classify([2],[1])
 def test_small_stationary_jitter_stays_one_pose_bucket(self):
  bins=PoseBuckets();a=np.eye(4);self.assertEqual(bins.index(0,a),0);b=a.copy();b[0,3]=.1;self.assertEqual(bins.index(0,b),0);b[0,3]=.6;self.assertEqual(bins.index(0,b),1)
 def test_rotation_and_camera_identity_can_create_distinct_geometry(self):
  bins=PoseBuckets();a=np.eye(4);bins.index(0,a);angle=np.radians(6);b=a.copy();b[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]];self.assertEqual(bins.index(0,b),1);self.assertEqual(bins.index(1,a),2)
 def test_region_area_is_exact_even_when_a_cell_intersects_both(self):
  body=footprint(np.eye(4),1.5,.8);ring=body.buffer(.5,resolution=32).difference(body);cells=cells_for_regions(body,ring,np.array([-10.,-10.]),.5,(50,50));self.assertAlmostEqual(sum(c['body_area_m2'] for c in cells),body.area);self.assertAlmostEqual(sum(c['ring_area_m2'] for c in cells),ring.area)
 def test_road_outer_shell_distinguishes_interior_holes_from_open_unknown(self):
  from shapely.geometry import Polygon,box
  from start_diagnostics import blind_area_topology
  road=Polygon([[-10,-10],[10,-10],[10,10],[-10,10]],holes=[[[-2,-2],[2,-2],[2,2],[-2,2]]]);body=box(-1,-1,1,1)
  self.assertTrue(blind_area_topology(road,body,body.buffer(.5))['body_inside_known_road_outer_shell']);self.assertFalse(road.covers(body))
  outside=box(11,-1,13,1);self.assertFalse(blind_area_topology(road,outside,outside.buffer(.5))['body_inside_known_road_outer_shell'])
if __name__=='__main__':unittest.main()
