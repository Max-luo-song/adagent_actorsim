"""CPU geometry tests for the pano lidar adapter (synthetic, no GPU)."""

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

import sys
import unittest
pass  # Path is provided by the portable boundary above

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pano_lidar import yaw_virtual_egos, pinhole_to_pano, PanoBuilder


class TestVirtualEgos(unittest.TestCase):
    def test_camera_lands_at_lidar_origin(self):
        lidar2world = np.eye(4); lidar2world[:3, 3] = [1.0, 2.0, 1.5]
        cam_to_ego = np.eye(4); cam_to_ego[:3, 3] = [0.5, 0.0, 0.2]
        egos = yaw_virtual_egos(lidar2world, cam_to_ego, 8)
        for k, ego in enumerate(egos):
            cam_world = ego @ cam_to_ego
            np.testing.assert_allclose(cam_world[:3, 3], [1.0, 2.0, 1.5], atol=1e-9)
        # 朝向均匀环绕
        yaws = [np.arctan2((e @ cam_to_ego)[1, 0], (e @ cam_to_ego)[0, 0]) for e in egos]
        self.assertEqual(len(np.unique(np.round(yaws, 6))), 8)

    def test_invalid_inputs(self):
        with self.assertRaises(Exception):
            yaw_virtual_egos(np.full((4, 4), np.nan), np.eye(4), 8)


class TestPanoGeometry(unittest.TestCase):
    def _wall_view(self, dist):
        """正前方一面墙：z=dist 均匀深度。"""
        H, W = 64, 128
        depth = np.full((H, W), dist, dtype=np.float32)
        K = np.array([[100., 0., W / 2], [0., 100., H / 2], [0., 0., 1.]])
        return depth, K

    def test_wall_distance(self):
        depth, K = self._wall_view(10.0)
        az, el, slant, valid = pinhole_to_pano(depth, K, np.eye(4),
                                               None, None)
        # 中心像素斜距==z；边缘更大
        mid = (depth.shape[0] // 2) * depth.shape[1] + depth.shape[1] // 2
        self.assertAlmostEqual(float(slant[mid]), 10.0, places=4)
        self.assertTrue(np.all(slant >= 10.0 - 1e-5))
        self.assertTrue(valid.all())

    def test_occlusion_zbuffer(self):
        builder = PanoBuilder()
        # 同一方向两次：远 10m 后近 5m → 保留 5m
        az = np.zeros(4); el = np.zeros(4)
        builder.add_view(az, el, np.full(4, 10.0), np.ones(4, bool))
        builder.add_view(az, el, np.full(4, 5.0), np.ones(4, bool))
        pts, valid = builder.to_points()
        r = np.linalg.norm(pts, axis=1)
        self.assertTrue(np.all(r <= 5.0 + 1e-4))

    def test_invalid_returns_excluded(self):
        builder = PanoBuilder()
        az = np.zeros(3); el = np.zeros(3)
        slant = np.array([10.0, np.nan, 0.0])
        valid = np.array([True, False, False])
        builder.add_view(az, el, slant, valid)
        pts, valid_mask = builder.to_points()
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(float(np.linalg.norm(pts[0])), 10.0, places=4)

    def test_seam_continuity(self):
        """az=+pi 与 -pi 应落在相邻网格（接缝不丢点）。"""
        builder = PanoBuilder(azimuth_res_deg=1.0)
        az = np.array([np.pi - 1e-6, -np.pi + 1e-6])
        el = np.zeros(2)
        builder.add_view(az, el, np.full(2, 8.0), np.ones(2, bool))
        pts, valid = builder.to_points()
        self.assertEqual(len(pts), 2)
        # ±pi 边界都在 -x 方向（后方）；两点应相邻（接缝不丢不裂）
        self.assertTrue(all(x < 0 for x in pts[:, 0]))
        self.assertLess(float(np.linalg.norm(pts[0] - pts[1])), 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
