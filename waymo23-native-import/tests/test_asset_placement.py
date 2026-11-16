import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from asset_placement import terrain_aligned_pose, supported_placement


class AssetPlacementTests(unittest.TestCase):
    def test_entire_support_plane_lies_on_sloped_ground_at_any_heading(self):
        fit = dict(origin_xy=[30000., -50000.], plane=[.07, -.04, 12.])
        for heading in (0., .7, -1.35):
            pose = terrain_aligned_pose([30002., -50001., 99.], heading, fit, .8)
            local = np.array([[x, .8, z, 1.] for x in (-2.2, 2.2) for z in (-.9, .9)])
            world = (local@pose.T)[:, :3]
            expected = (world[:, :2]-fit['origin_xy'])@np.asarray(fit['plane'][:2])+12.
            np.testing.assert_allclose(world[:, 2], expected, atol=1e-9)
            np.testing.assert_allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)

    def test_flat_ground_preserves_library_axis_convention(self):
        pose = terrain_aligned_pose([1, 2, 9], 0, dict(origin_xy=[0, 0], plane=[0, 0, 3]), .8)
        np.testing.assert_allclose(pose[:3, :3], [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_allclose(pose[:3, 3], [1, 2, 3.8])

    def test_shared_search_rejects_occupied_anchor_and_selects_supported_neighbor(self):
        xy = np.stack(np.meshgrid(np.arange(-5, 5.01, .2), np.arange(-5, 5.01, .2)), axis=-1).reshape(-1, 2)
        skin = dict(ground_points=np.column_stack([xy, np.zeros(len(xy))]),
                    occupied_points=np.array([[0., 0., .5]]), source_kind='sensor', source_id='synthetic-test')
        pose, report = supported_placement([0, 0, 0], 0., dict(origin_xy=[0, 0], plane=[0, 0, 0]), 0., [1., 2., 1.], skin)
        self.assertIn('intersects', report['trials'][0]['rejection'])
        self.assertNotEqual(report['selected_offset_m'], [0, 0])
        self.assertFalse(report['real_wheel_contacts_verified'])
        self.assertFalse(report['n1_edit_certified'])
