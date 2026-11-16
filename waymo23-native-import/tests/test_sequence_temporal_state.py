"""Regression for fresh UniAD without prev_bev and preservation after inference."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE/"waymo23-native-import"),
               str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad")]
from run_reliable_sequence import synchronize_temporal_state, prepare_temporal_frame
from scene_reliability import SceneReliability


class TemporalStateTests(unittest.TestCase):
    def test_scene_switch_accepts_new_clock_and_discards_all_history(self):
        core = self.core()
        self.assertTrue(prepare_temporal_frame(core, "scene-a", 12.))
        bev = object()
        core.prev_bev, core.test_track_instances = bev, object()
        self.assertFalse(prepare_temporal_frame(core, "scene-a", 12.1, .1))
        self.assertIs(core.prev_bev, bev)
        core.scene_reliability.tracks = [{"id": 1}]
        self.assertTrue(prepare_temporal_frame(core, "unseen-scene-b", 0., .1))
        self.assertIsNone(core.prev_bev)
        self.assertIsNone(core.test_track_instances)
        self.assertEqual(core.scene_reliability.tracks, [])

    def test_gap_resets_and_bad_clock_rejected(self):
        core = self.core()
        prepare_temporal_frame(core, "a", 0.)
        self.assertTrue(prepare_temporal_frame(core, "a", 2., .1))
        for stamp in (2., 1., float("nan")):
            with self.assertRaises(ValueError):
                prepare_temporal_frame(core, "a", stamp, .1)
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            prepare_temporal_frame(core, "a", 2.2, .1)

    def test_explicit_restart_can_replay_same_scene_from_earlier_clock(self):
        core = self.core()
        prepare_temporal_frame(core, "a", 10.)
        prepare_temporal_frame(core, "a", 10.5)
        synchronize_temporal_state(core, reset=True)
        self.assertTrue(prepare_temporal_frame(core, "a", 10.))
        self.assertEqual(core._adapter_timestamp, 10.)

    def core(self):
        return SimpleNamespace(prev_frame_info={"prev_bev": None},
                               test_track_instances=None, scene_reliability=SceneReliability())

    def test_first_frame_without_lazy_bev_attribute(self):
        core = self.core()
        synchronize_temporal_state(core)
        self.assertIsNone(core.prev_frame_info["prev_bev"])
        synchronize_temporal_state(core, reset=True)
        self.assertIsNone(core.prev_bev)

    def test_next_frame_preserves_inferred_bev_and_tracks(self):
        core = self.core()
        synchronize_temporal_state(core, reset=True)
        bev, tracks = object(), object()
        core.prev_bev, core.test_track_instances = bev, tracks
        synchronize_temporal_state(core)
        self.assertIs(core.prev_frame_info["prev_bev"], bev)
        self.assertIs(core.test_track_instances, tracks)

    def test_gap_reset_discards_neural_and_evidence_history(self):
        core = self.core()
        core.prev_bev, core.test_track_instances = object(), object()
        core.scene_reliability.previous_time = 10.
        core.scene_reliability.tracks = [{"id": 1}]
        synchronize_temporal_state(core, reset=True)
        self.assertIsNone(core.prev_bev)
        self.assertIsNone(core.prev_frame_info["prev_bev"])
        self.assertIsNone(core.test_track_instances)
        self.assertIsNone(core.scene_reliability.previous_time)
        self.assertEqual(core.scene_reliability.tracks, [])


if __name__ == "__main__":
    unittest.main()
