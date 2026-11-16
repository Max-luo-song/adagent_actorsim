
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

import copy,pickle,sys,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from global_repair_v2.unknown_recovery import TemporalOccupancy,RecoveryDepthMemory,OriginalDepthObstacleMemory
from runtime_configuration import load_vehicle_config

class UnknownRecoveryTests(unittest.TestCase):
 def fixture(self,known_point=False,occupied_inside=False,unknown_inside=False):
  v=load_vehicle_config();m=np.zeros((1,5,200,200));m[:,:,190,190]=1
  if occupied_inside:m[:,:,100,100]=1
  center=[0.,0.] if unknown_inside else [v.half_width_m+.35,0.]
  t=TemporalOccupancy(m,v,unknown_points=[center],unknown_cell_rotation=np.eye(3),unknown_point_radius=np.sqrt(3)*.25+.245,observed_points=[[0.,0.]] if known_point else [],observed_cell_rotation=np.eye(3),observed_point_radius=np.sqrt(3)*.25)
  initial=np.zeros(5);gap=float(t.unknown_gap(initial[None])[0]);info=dict(scene_token='test',timestamp=0.,model_lidar_frame=dict(virtual_to_ego=np.eye(4)),explicit_l2g_r_mat=np.eye(3),ego_velo=0.,ego_steer=0.)
  episode=dict(scene_token='test',start=0.,deadline=3.,initial_gap_m=gap,timestamp=0.)
  rel=dict(unknown_visibility_geometry=dict(initial_unknown_margin_recovery=dict(episode=episode)))
  return v,t,initial,info,rel
 def test_only_margin_can_restore_and_original_voxels_stay_hard(self):
  _,t,s,i,r=self.fixture();t.configure_recovery(s,i,r,1.)
  self.assertIsNotNone(t.recovery);self.assertGreater(t.unknown_gap(s[None],True)[0],0)
  self.assertAlmostEqual(t.clearance(s[None],[0])[0],0.,places=8)
  inside=s.copy();inside[0]=1.;self.assertLess(t.clearance(inside[None],[0])[0],0)
 def test_staying_still_fails_progress_and_full_margin_required_at_deadline(self):
  _,t,s,i,r=self.fixture();t.configure_recovery(s,i,r,1.)
  self.assertLess(t.clearance(s[None],[.1])[0],0)
  escaped=s.copy();escaped[0]=-1.
  self.assertGreaterEqual(t.clearance(escaped[None],[3.])[0],0)
  self.assertGreaterEqual(t.unknown_gap(escaped[None])[0],0)
 def test_measured_occupied_or_unknown_cell_overlap_never_relaxes(self):
  for kw in (dict(known_point=True),dict(occupied_inside=True),dict(unknown_inside=True)):
   _,t,s,i,r=self.fixture(**kw);expected=t.clearance(s[None],[0]).copy();t.configure_recovery(s,i,r,1.)
   self.assertIsNone(t.recovery);np.testing.assert_array_equal(t.clearance(s[None],[0]),expected)
 def test_proposal_conflict_never_relaxes(self):
  _,t,s,i,r=self.fixture();t.configure_recovery(s,i,r,-.1);self.assertIsNone(t.recovery)
 def test_no_state_preserves_original_temporal_clearance(self):
  _,t,s,i,r=self.fixture();expected=super(TemporalOccupancy,t).clearance(s[None],[0]);t.configure_recovery(s,i,{},1.)
  np.testing.assert_array_equal(t.clearance(s[None],[0]),expected)
 def test_forecast_fallback_cannot_freeze_restoration_clock(self):
  _,t,s,i,r=self.fixture();t.configure_recovery(s,i,r,1.)
  self.assertLess(t.clearance(s[None],[0.],unknown_times=[1.])[0],0)
 def test_stale_or_malformed_episode_rejected(self):
  _,t,s,i,r=self.fixture()
  for change in (dict(timestamp=.1),dict(deadline=30.),dict(initial_gap_m=float('nan'))):
   invalid=copy.deepcopy(r);invalid['unknown_visibility_geometry']['initial_unknown_margin_recovery']['episode'].update(change)
   with self.assertRaises(ValueError):t.configure_recovery(s,i,invalid,1.)
 def test_episode_deadline_survives_replanning_and_pickle(self):
  v,_,s,info,_=self.fixture();center=[v.half_width_m+.35,0.]
  def update(self,obs,info):return np.empty((0,2)),dict(unknown_visibility_points_xy=[center],unknown_visibility_geometry=dict(point_radius_m=np.sqrt(3)*.25+.245))
  with patch.object(OriginalDepthObstacleMemory,'update',update):
   memory=RecoveryDepthMemory(v);memory.update({},info);deadline=memory.recovery_episode['deadline'];blob=pickle.dumps(memory)
   memory=pickle.loads(blob);_,audit=memory.update({},dict(info,timestamp=.1));self.assertEqual(memory.recovery_episode['deadline'],deadline)
   _,audit=memory.update({},dict(info,timestamp=3.1));self.assertEqual(memory.recovery_episode['deadline'],deadline)
   self.assertFalse(audit['unknown_visibility_geometry']['initial_unknown_margin_recovery']['free_space_asserted'])
   memory.update({},dict(info,scene_token='new',timestamp=0.));self.assertEqual(memory.recovery_episode['scene_token'],'new')
if __name__=='__main__':unittest.main()
