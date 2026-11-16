"""CPU regressions for episode resets using the worker's real temporal adapter."""

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

pass  # Path is provided by the portable boundary above
import sys
from types import SimpleNamespace
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'waymo23-native-import').is_dir())
sys.path.insert(0,str(ROOT/'waymo23-native-import'))
from resident_session_state import ResidentSessionState


class Parameter:
    def data_ptr(self):return 12345


class Reliability:
    def __init__(self):
        self.config=SimpleNamespace(max_gap_seconds=.75)
        self.reset()
    def reset(self):
        self.tracks=[];self.next_id=0;self.scene_token=None;self.previous_time=None


class Core:
    def __init__(self):
        self.parameter=Parameter()
        self.prev_bev=None;self.test_track_instances=None
        self.scene_reliability=Reliability()
        self.prev_frame_info=dict(prev_bev=None,scene_token=None,prev_pos=0,prev_angle=0)
        self.track_base=SimpleNamespace(max_obj_id=0,score_thresh=.4)
        self.track_base.clear=lambda:setattr(self.track_base,'max_obj_id',0)
        self.l2g_r_mat=self.l2g_t=self.timestamp=self.scene_token=None
    def parameters(self):return iter([self.parameter])
    def infer(self):
        self.prev_bev=object();self.test_track_instances=[object()]
        self.prev_frame_info.update(prev_bev=self.prev_bev,scene_token='old',prev_pos=[1,2,3],prev_angle=20)
        self.track_base.max_obj_id=55
        self.scene_reliability.tracks=[object()];self.scene_reliability.next_id=32
        self.scene_reliability.scene_token='old';self.scene_reliability.previous_time=8.1
        self.l2g_r_mat=self.l2g_t=self.timestamp=object()


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.core=Core();self.session=ResidentSessionState(self.core)
    def test_two_episodes_clear_bev_tracks_ids_pose_and_evidence(self):
        first=self.session.prepare('552-first',8.,.1)
        self.assertTrue(first['reset_state_cleared'])
        self.core.infer()
        second=self.session.prepare('552-reset',8.,.1)
        self.assertTrue(second['before']['bev_present'])
        self.assertTrue(second['before']['tracking_present'])
        self.assertTrue(second['reset_state_cleared'])
        self.assertEqual(second['identity'],first['identity'])
        self.assertEqual(self.core.prev_frame_info['prev_pos'],0)
        self.assertEqual(self.core.track_base.score_thresh,.4)
    def test_within_episode_preserves_model_history(self):
        self.session.prepare('552',8.,.1);self.core.infer()
        bev=self.core.prev_bev;tracks=self.core.test_track_instances
        audit=self.session.prepare('552',8.1,.1)
        self.assertFalse(audit['reset'])
        self.assertIs(self.core.prev_bev,bev)
        self.assertIs(self.core.test_track_instances,tracks)
        self.assertEqual(self.core.track_base.max_obj_id,55)
    def test_different_scene_then_return_each_start_clean(self):
        for token in ('552-first','551','552-return'):
            audit=self.session.prepare(token,8.,.1)
            self.assertTrue(audit['reset_state_cleared'])
            self.core.infer()
        self.assertEqual(audit['reset_count'],3)
    def test_time_gap_resets_but_nonmonotone_same_episode_rejects(self):
        self.session.prepare('552',8.,.1);self.core.infer()
        with self.assertRaises(ValueError):self.session.prepare('552',8.,.1)
        self.assertTrue(self.session.prepare('552',9.,.1)['reset_state_cleared'])
    def test_invalid_context_does_not_destroy_old_state(self):
        self.session.prepare('552',8.,.1);self.core.infer()
        bev=self.core.prev_bev
        for token,stamp,dt in [('',8.1,.1),('new',float('nan'),.1),('new',8.1,0)]:
            with self.assertRaises(ValueError):self.session.prepare(token,stamp,dt)
        self.assertIs(self.core.prev_bev,bev)
    def test_failed_track_counter_reset_rejects_frame(self):
        self.core.track_base.max_obj_id=99;self.core.track_base.clear=lambda:None
        with self.assertRaisesRegex(RuntimeError,'left temporal state behind'):
            self.session.prepare('552',8.,.1)
    def test_replacement_model_parameter_is_detected(self):
        self.core.parameter=Parameter()
        with self.assertRaisesRegex(RuntimeError,'identity changed'):
            self.session.prepare('552',8.,.1)


if __name__=='__main__':unittest.main()
