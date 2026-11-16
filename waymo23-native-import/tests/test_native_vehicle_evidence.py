"""Sparse diagnostic evidence cannot silently become dense visibility truth."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_native_vehicle_evidence import (CandidateAuditPolicy, box_membership,
                                           configured_holdout_neighbors, sparse_view_support)


class NativeEvidenceTests(unittest.TestCase):
    def test_holdout_neighbors_follow_one_rule_at_boundary_and_exact_frame(self):
        self.assertEqual(configured_holdout_neighbors(146, 198, 10), [140, 150])
        self.assertEqual(configured_holdout_neighbors(60, 198, 10), [50, 60, 70])
        self.assertEqual(configured_holdout_neighbors(0, 20, 10), [10])
        self.assertEqual(configured_holdout_neighbors(19, 20, 10), [10])
        self.assertEqual(configured_holdout_neighbors(19, 20, 0), [])

    def test_object_membership_respects_pose_and_rejects_invalid_geometry(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [100, -80, 2]
        local = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 0]])
        world = local@pose[:3, :3].T+pose[:3, 3]
        np.testing.assert_array_equal(box_membership(world, pose, [4, 2, 2]), [True, False, True])
        pose[0, 0] = 2
        with self.assertRaises(ValueError):
            box_membership(world, pose, [4, 2, 2])

    def test_closer_competing_lidar_rejects_sparse_support(self):
        policy = CandidateAuditPolicy(minimum_points_per_view=1, minimum_cells_per_view=1)
        uv = np.array([[10., 10.], [11., 11.], [100, 100]])
        target = np.array([True, False, False])
        report, support = sparse_view_support(uv, np.array([10., 5., 2.]), target, target, [64, 64], policy)
        self.assertEqual(report['contradicted_points'], 1)
        self.assertEqual(len(support), 0)
        self.assertFalse(report['selected_for_diagnostic'])

    def test_absence_of_occluder_remains_unknown_and_ambiguous_points_excluded(self):
        policy = CandidateAuditPolicy(minimum_points_per_view=1, minimum_cells_per_view=1)
        uv = np.array([[10., 10.], [30., 30.]])
        target = np.array([True, True])
        eligible = np.array([True, False])
        report, support = sparse_view_support(uv, np.array([10., 10.]), target, eligible, [64, 64], policy)
        np.testing.assert_array_equal(support, [0])
        self.assertTrue(report['selected_for_diagnostic'])
        self.assertFalse(report['visibility_proven'])
        self.assertFalse(report['dense_foreground_mask_available'])

    def test_points_behind_camera_or_outside_image_never_support_a_view(self):
        uv = np.array([[10., 10.], [-1., 10.], [64., 10.], [np.nan, 10.]])
        mask = np.ones(4, dtype=bool)
        report, _ = sparse_view_support(uv, np.array([-1., 10., 10., 10.]), mask, mask,
                                         [64, 64], CandidateAuditPolicy())
        self.assertEqual(report['projected_eligible_points'], 0)


if __name__ == '__main__':
    unittest.main()
