
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
from effective_start_scan import trajectory_stats,windows
class EffectiveStartTests(unittest.TestCase):
 def test_exact_last_frame_budget_includes_final_observation(self):
  arc=np.arange(198,dtype=float);gaps=np.ones(198)
  self.assertIn(97,windows(arc,gaps,100)['full_window_starts']);self.assertNotIn(98,windows(arc,gaps,100)['full_window_starts'])
 def test_late_geometric_start_does_not_become_a_complete_episode(self):
  gaps=np.full(198,-1.);gaps[162]=.1101;r=windows(np.arange(198,dtype=float),gaps,100)
  self.assertEqual(r['first_footprint_candidate'],162);self.assertIsNone(r['first_full_window_start'])
 def test_earliest_rule_uses_no_controller_score(self):
  gaps=np.full(198,-1.);gaps[[50,80,100]]=.1;r=windows(np.arange(198,dtype=float)/10,gaps,100,6.)
  self.assertEqual(r['first_displacement_candidate'],60);self.assertEqual(r['first_full_window_start'],80)
 def test_arc_length_is_distinct_from_net_forward_distance(self):
  poses=np.tile(np.eye(4),(3,1,1));poses[1,0,3]=4;poses[2,0,3]=0
  arc,offset,forward=trajectory_stats(poses);self.assertEqual(arc[-1],8);self.assertEqual(offset[-1],0);self.assertEqual(forward[-1],0)
 def test_missing_map_cannot_claim_eligibility(self):
  r=windows(np.arange(198,dtype=float),np.full(198,np.nan));self.assertIsNone(r['first_full_window_start'])
if __name__=='__main__':unittest.main()
