"""Compare against the installed legacy library, not hand-copied expectations."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import box_corners, box_footprint, points_to_box_xy, assert_box_class_contract
from scene_reliability import ReliabilityConfig, path_clearance, augment_occupancy


def installed_corners_oracle(boxes):
    # Only pure tensor functions are extracted; no compiled GPU ops imported.
    import torch
    folder = BASE/'code/mmdetection3d-uniad-legacy/mmdet3d/core/bbox/structures'
    utils = ast.parse((folder/'utils.py').read_text())
    rotation = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == 'rotation_3d_in_axis')
    tree = ast.parse((folder/'lidar_box3d.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LiDARInstance3DBoxes')
    corners = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'corners')
    corners.decorator_list = []
    scope = dict(torch=torch, np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rotation, corners], type_ignores=[])),
                 str(folder/'lidar_box3d.py'), 'exec'), scope)
    tensor = torch.as_tensor(np.asarray(boxes), dtype=torch.float64)
    return scope['corners'](SimpleNamespace(tensor=tensor, dims=tensor[:, 3:6])).numpy()


class BoxGeometryContractTests(unittest.TestCase):
    def test_runtime_check_rejects_changed_dependency_rotation(self):
        class Installed:
            def __init__(self, data, box_dim):
                self.data = data
            @property
            def corners(self):
                return installed_corners_oracle(self.data)
        class Incompatible(Installed):
            @property
            def corners(self):
                boxes = np.asarray(self.data).copy()
                boxes[:, 6] *= -1
                return installed_corners_oracle(boxes)
        self.assertTrue(assert_box_class_contract(Installed, np.asarray)['installed_box_class_verified'])
        with self.assertRaisesRegex(ValueError, 'differs'):
            assert_box_class_contract(Incompatible, np.asarray)
    def test_random_asymmetric_boxes_match_installed_legacy_corners(self):
        rng = np.random.default_rng(20261020)
        boxes = np.column_stack([rng.uniform(-100, 100, (60, 3)),
                                 rng.uniform(.3, 9, (60, 3)), rng.uniform(-np.pi, np.pi, 60)])
        expected = installed_corners_oracle(boxes)
        actual = np.stack([box_corners(box) for box in boxes])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_rotation_roundtrip_uses_clockwise_legacy_yaw(self):
        box = np.array([3., 8., -1., 1., 8., 2., np.pi/4])
        local = points_to_box_xy(box_footprint(box), box)
        np.testing.assert_allclose(local, [[-.5, -4], [.5, -4], [.5, 4], [-.5, 4]], atol=1e-12)

    def test_rotated_obstacle_on_path_is_not_mirrored_away(self):
        box = np.array([3., 8., 0., 1., 8., 2., np.pi/4, 0., 0.])
        path = np.column_stack([np.zeros(6), np.arange(1, 7)])
        cfg = ReliabilityConfig(ego_half_width_m=.2, ego_half_length_m=.2, obstacle_margin_m=.1)
        self.assertLess(path_clearance(box, path, cfg), 0.)
        mirrored = box.copy()
        mirrored[6] *= -1
        self.assertGreater(path_clearance(mirrored, path, cfg), 0.)

    def test_occupancy_marks_true_diagonal_and_preserves_raw_occupancy(self):
        raw = np.zeros((1, 5, 200, 200))
        report = dict(obstacles=[dict(state='provisional_uncertain', box=[3.25, 8.25, 0., 1., 8., 2., np.pi/4, 0., 0.])])
        guarded = augment_occupancy(raw, report)
        self.assertTrue(guarded[0, :, 110, 100].all())  # (0.25,5.25), true clockwise long axis
        self.assertFalse(guarded[0, :, 122, 100].any())  # (0.25,11.25), erroneous mirrored axis
        self.assertFalse(raw.any())

    def test_clearance_is_equivariant_to_unseen_global_headings(self):
        box = np.array([3., 8., 0., 1., 8., 2., .7, .1, .2])
        path = np.column_stack([np.zeros(6), np.arange(1, 7)])
        cfg = ReliabilityConfig()
        expected = path_clearance(box, path, cfg)
        for angle in (.2, -1.3, 2.8):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, s], [-s, c]])
            transformed = box.copy()
            transformed[:2] = box[:2]@rotation
            transformed[7:9] = box[7:9]@rotation
            transformed[6] -= angle
            self.assertAlmostEqual(path_clearance(transformed, path@rotation, cfg), expected, places=10)
