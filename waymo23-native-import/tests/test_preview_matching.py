from pathlib import Path
import sys
import unittest
import json
import pickle
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_preview_vehicle import polygon_iou, box_polygon, verify_prediction_binding, evaluate
from perception_run_binding import file_fingerprint


class PreviewMatchingTests(unittest.TestCase):
    def test_rotated_polygon_identity_disjoint_and_partial_overlap(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2.]])
        self.assertAlmostEqual(polygon_iou(square, square[::-1]), 1.)
        self.assertEqual(polygon_iou(square, square+10), 0.)
        self.assertAlmostEqual(polygon_iou(square, square+[1, 0]), 1/3)
        c, s = np.cos(.7), np.sin(.7)
        rotation = np.array([[c, -s], [s, c]])
        self.assertAlmostEqual(polygon_iou(square@rotation, (square+[1, 0])@rotation), 1/3)

    def test_box_yaw_rotates_width_and_length(self):
        p = box_polygon(np.array([0, 0, 0, 2, 4, 1, np.pi/2]))
        np.testing.assert_allclose(np.ptp(p, axis=0), [4, 2])


class PredictionBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obs = dict(rgb=dict(CAM_FRONT=np.zeros((2, 2, 3), dtype=np.uint8)))
        self.info = dict(explicit_l2g_r_mat=np.eye(3), explicit_l2g_t=np.zeros(3),
                         timestamp=0., command=2, ego_velo=1., ego_steer=0.,
                         cam_params=dict(CAM_FRONT=dict(intrinsic=dict(W=2, H=2), l2c=np.eye(4))))
        self.packet = self.root/'packet.pkl'
        self.save(self.packet, self.info)
        self.sequence = self.root/'sequence.json'
        self.sequence.write_text(json.dumps(dict(frames=[dict(frame=0, timestamp=0., packet=str(self.packet))])))
        self.run = dict(source=str(self.sequence), completed_frames=1)
        self.write_run()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, path, info):
        with path.open('wb') as stream:
            pickle.dump((self.obs, info), stream)

    def write_run(self):
        (self.root/'run_report.json').write_text(json.dumps(self.run))

    def test_same_packets_accept_equivalent_opt_in_manifest(self):
        preview = self.root/'preview.json'
        data = json.loads(self.sequence.read_text())
        data.update(diagnostic_preview_only=True, perception_preview_requested=True)
        preview.write_text(json.dumps(data))
        verify_prediction_binding(preview, preview, self.root)

    def test_partial_run_does_not_become_a_complete_matching_report(self):
        self.run['completed_frames'] = 0
        self.write_run()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            verify_prediction_binding(self.sequence, self.sequence, self.root)

    def test_predictions_from_other_packet_are_rejected(self):
        other = self.root/'other.json'
        data = json.loads(self.sequence.read_text())
        data['frames'][0]['packet'] = str(self.root/'other.pkl')
        other.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            verify_prediction_binding(self.sequence, other, self.root)

    def test_paired_baseline_with_different_camera_pose_rejected(self):
        import copy
        info = copy.deepcopy(self.info)
        info['cam_params']['CAM_FRONT']['l2c'][0, 3] = 1.
        other_packet = self.root/'other.pkl'
        self.save(other_packet, info)
        other = self.root/'other.json'
        data = json.loads(self.sequence.read_text())
        data['frames'][0]['packet'] = str(other_packet)
        other.write_text(json.dumps(data))
        self.run['source'] = str(other)
        self.write_run()
        with self.assertRaisesRegex(ValueError, 'calibration differs'):
            verify_prediction_binding(self.sequence, other, self.root)

    def test_missing_prediction_does_not_silently_truncate(self):
        asset = self.root/'asset.json'
        asset.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Missing prediction frame'):
            evaluate(self.sequence, asset, self.root, self.root/'report.json')
        self.assertFalse((self.root/'report.json').exists())

    def test_input_content_change_invalidates_recorded_binding(self):
        self.run.update(input_binding=dict(sequence=file_fingerprint(self.sequence),
            packets=[dict(frame=0, timestamp=0., **file_fingerprint(self.packet))]),
            implementation_unchanged_during_run=True)
        self.write_run()
        with self.packet.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'packet content changed'):
            verify_prediction_binding(self.sequence, self.sequence, self.root)
