from pathlib import Path
import sys
import unittest

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE/'waymo23-native-import'),
               str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from perception_preview import preview_requested, QuarantinedPerceptionObserver
from run_reliable_sequence import inspect_packet, SLOTS
from edit_trust import assert_edit_evaluable, EditTrustError
from build_source_sequence import preview_opt_in


class PerceptionPreviewTests(unittest.TestCase):
    def test_rendered_diagnostic_requires_explicit_typed_opt_in(self):
        self.assertFalse(preview_opt_in(dict(diagnostic_asset={'asset': 'x'})))
        self.assertTrue(preview_opt_in(dict(diagnostic_asset={'asset': 'x'}, perception_preview_requested=True)))
        for source in (dict(perception_preview_requested=True),
                       dict(diagnostic_asset={'asset': 'x'}, perception_preview_requested='true')):
            with self.assertRaises(ValueError):
                preview_opt_in(source)

    def test_opt_in_is_explicit_and_diagnostic_only(self):
        self.assertFalse(preview_requested(dict(diagnostic_preview_only=True)))
        self.assertTrue(preview_requested(dict(diagnostic_preview_only=True, perception_preview_requested=True)))
        for value in [dict(perception_preview_requested=True), dict(perception_preview_requested='true')]:
            with self.assertRaises(ValueError):
                preview_requested(value)

    def test_quarantined_observer_never_exports_control_or_trust(self):
        report = QuarantinedPerceptionObserver().update([], [], [])
        self.assertFalse(report['evaluation_admitted'])
        self.assertFalse(report['edit_trust_verified'])
        self.assertTrue(report['control_use_forbidden'])
        self.assertNotIn('action', report)
        self.assertNotIn('speed_limit_mps', report)

    def test_real_edited_packet_stays_blocked_for_normal_inference(self):
        packet = BASE/'outputs/gpu_test_service/jobs/n1-library-172-015102/result/frame_146/observation.pkl'
        if not packet.exists():
            self.skipTest('Saved GPU fixture unavailable')
        with self.assertRaises(EditTrustError):
            inspect_packet(packet, SLOTS)
        obs, info, *rest = inspect_packet(packet, SLOTS, perception_preview=True)
        self.assertTrue(info['edited_scene'])
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable(info)
