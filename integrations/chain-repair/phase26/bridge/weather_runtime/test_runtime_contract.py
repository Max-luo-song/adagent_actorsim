"""Sensor alignment and actor preservation across the weather RGB boundary."""
import unittest
import numpy as np
from runtime_contract import CAMERAS, prepare_triplet, integrate_triplet


class RuntimeContractTest(unittest.TestCase):
    def make_raw(self):
        cameras = {}
        for i, name in enumerate(CAMERAS):
            rgb = np.full((12, 18, 3), 20 + 40 * i, dtype=np.uint8)
            mask = np.zeros((12, 18), dtype=np.uint8)
            mask[3:8, 4:9] = 255
            cameras[name] = dict(rgb=rgb, weather_dynamic_mask=mask,
                depth=np.ones((12, 18), dtype=np.float32), K=np.eye(3),
                camera_to_world=np.eye(4), source_camera_id=i)
        # Deliberately different dict order; explicit camera order must win.
        return dict(cameras=dict(reversed(list(cameras.items()))))

    def test_camera_order_and_mask_alignment(self):
        rgb, mask = prepare_triplet(self.make_raw())
        self.assertEqual(rgb.shape, (3, 256, 384, 3))
        self.assertEqual(mask.shape, (3, 256, 384))
        self.assertEqual(rgb[:, 0, 0, 0].tolist(), [20, 60, 100])
        self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_full_resolution_actor_protection_and_geometry(self):
        raw = self.make_raw()
        result = integrate_triplet(raw, np.full((3, 256, 384, 3), 200, dtype=np.uint8))
        for name in CAMERAS:
            old, new = raw['cameras'][name], result['cameras'][name]
            actor = old['weather_dynamic_mask'] > 0
            np.testing.assert_array_equal(new['rgb'][actor], old['rgb'][actor])
            self.assertTrue(np.all(new['rgb'][~actor] == 200))
            self.assertTrue(np.all(old['rgb'] == old['rgb'][0, 0]))
            for key in ('depth', 'K', 'camera_to_world'):
                self.assertIs(new[key], old[key])
            self.assertEqual(new['rgb'].shape, old['rgb'].shape)

    def test_invalid_predictions_and_masks_rejected(self):
        raw = self.make_raw()
        with self.assertRaises(ValueError):
            integrate_triplet(raw, np.zeros((3, 256, 384, 3)))
        raw['cameras']['CAM_FRONT']['weather_dynamic_mask'] = np.zeros((3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            prepare_triplet(raw)


if __name__ == '__main__':
    unittest.main()
