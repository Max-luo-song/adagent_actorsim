import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_n1_sensor_evidence import project, CV_TO_WAYMO, projection_on_optical_branch


class SensorProjectionTests(unittest.TestCase):
    def test_folded_off_axis_ray_inside_image_is_rejected(self):
        intrinsic = np.array([2055.556, 2055.556, 939.657, 641.072,
                              .032316, -.321412, .000793, -.000626, 0.])
        points = np.array([[1.308, .212, 1.], [.1, .1, 1.]])
        pixels, depth = project(points, np.eye(4), intrinsic)
        self.assertTrue(((pixels[0] > 0) & (pixels[0] < [1920, 1280])).all())
        np.testing.assert_array_equal(projection_on_optical_branch(
            points, np.eye(4), intrinsic, pixels, depth), [False, True])

    def test_optical_branch_preserves_valid_tangential_distortion_and_pose(self):
        intrinsic = np.array([100., 100., 320., 240., .1, -.01, .001, -.002, 0.])
        camera = np.eye(4)
        camera[:3, 3] = [10., 20., 3.]
        points = np.array([[10.2, 20.1, 4.], [10., 20., 2.]])
        np.testing.assert_array_equal(projection_on_optical_branch(points, camera, intrinsic), [True, False])

    def test_waymo_forward_and_left_axes(self):
        intrinsic = np.array([100., 100., 320., 240., 0, 0, 0, 0, 0])
        pixels, depth = project(np.array([[10., 0, 0], [10., 1, 1]]), CV_TO_WAYMO, intrinsic)
        np.testing.assert_allclose(pixels, [[320, 240], [310, 230]])
        np.testing.assert_allclose(depth, [10, 10])

    def test_distortion_and_camera_pose(self):
        camera = np.eye(4)
        camera[:3, 3] = [3, 4, 5]
        intr = np.array([100., 100., 0, 0, .1, 0, 0, 0, 0])
        pixels, depth = project(np.array([[4., 4., 7.]]), camera, intr)
        np.testing.assert_allclose(pixels, [[51.25, 0]])
        np.testing.assert_allclose(depth, [2])
