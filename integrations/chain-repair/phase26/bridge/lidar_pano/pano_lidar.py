"""Pano-depth LiDAR approximation on the verified MultiTrainer renderer.

诊断、未认证。**明确标注：这是近似 LiDAR**——相机渲染器的多视锥柱面拼接，
不是原生 lidargs 扫描渲染，不宣称等价。

链路：K 个 pinhole 视锥环绕 360°（同一场景/时间/编辑状态的 MultiTrainer 渲染）
→ 逐像素投到柱面 (azimuth, elevation) 网格 → z-buffer 取最近
→ pano_to_lidar_with_intensities 的纯几何部分转点云（**不带强度**：
无可靠强度依据时不输出伪强度字段）。

坐标：全部在 lidar 系（输入渲染世界系位姿，内部转 lidar 系）。
"""
from __future__ import annotations

import numpy as np


def yaw_virtual_egos(lidar_to_world, cam_to_ego, n_views):
    """为每个 yaw 步进构造虚拟 ego 位姿，使相机落在 lidar 原点朝各方向。

    cam_world = ego_world @ cam_to_ego ⇒ 目标 cam_world = lidar_world @ Rz(yaw)
    ⇒ ego_world = lidar_world @ Rz(yaw) @ inv(cam_to_ego)
    """
    lidar_to_world = np.asarray(lidar_to_world, dtype=float)
    cam_to_ego = np.asarray(cam_to_ego, dtype=float)
    if not np.isfinite(lidar_to_world).all() or not np.isfinite(cam_to_ego).all():
        raise ValueError('non-finite transform')
    if n_views < 1:
        raise ValueError('n_views must be >= 1')
    inv_c2e = np.linalg.inv(cam_to_ego)
    out = []
    for k in range(n_views):
        yaw = 2.0 * np.pi * k / n_views
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        out.append(lidar_to_world @ Rz @ inv_c2e)
    return out


def pinhole_to_pano(depth, K, cam_to_lidar, azimuth_grid, elevation_grid):
    """把单视锥深度图投到柱面网格；返回 (pano_depth, pano_mask) 由调用方 z-buffer。

    depth: (H,W) 米制射线深度（框架渲染为相机 z 深度——本函数按 z 深度处理，
    调用方负责确认；转换: 斜距 = z / cos(光线与光轴夹角)）。
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # 相机系（opencv: x右 y下 z前）→ 斜距
    slant = np.sqrt(x * x + y * y + z * z)
    pts_cam = np.stack([x, y, z], -1).reshape(-1, 3)
    valid = (z.reshape(-1) > 1e-3) & np.isfinite(z.reshape(-1))
    # 相机系 → lidar 系
    pts_l = (np.asarray(cam_to_lidar)[:3, :3] @ pts_cam.T).T + np.asarray(cam_to_lidar)[:3, 3]
    az = np.arctan2(pts_l[:, 1], pts_l[:, 0])          # [-pi, pi]
    el = np.arcsin(np.clip(pts_l[:, 2] / np.linalg.norm(pts_l, axis=1), -1, 1))
    return az, el, slant.reshape(-1), valid


class PanoBuilder:
    """柱面网格 z-buffer 合成。"""

    def __init__(self, azimuth_res_deg=0.2, elev_min_deg=-25.0, elev_max_deg=15.0,
                 elev_res_deg=0.4, max_range=200.0):
        self.az = np.arange(-np.pi, np.pi, np.deg2rad(azimuth_res_deg))
        self.el = np.arange(np.deg2rad(elev_min_deg), np.deg2rad(elev_max_deg),
                            np.deg2rad(elev_res_deg))
        self.max_range = float(max_range)
        self.depth = np.full((len(self.el), len(self.az)), np.inf, dtype=np.float32)

    def add_view(self, az, el, slant, valid):
        iaz = np.digitize(az, self.az) - 1
        iel = np.digitize(el, self.el) - 1
        ok = (iaz >= 0) & (iaz < len(self.az)) & (iel >= 0) & (iel < len(self.el)) & valid
        for a, e, r in zip(iaz[ok], iel[ok], slant[ok]):
            if r < self.depth[e, a]:
                self.depth[e, a] = r

    def to_points(self):
        """pano 深度 → lidar 系点云（xyz + 有效性）。无强度字段。"""
        el_g, az_g = np.meshgrid(self.el, self.az, indexing='ij')
        valid = np.isfinite(self.depth) & (self.depth < self.max_range)
        r = np.where(valid, self.depth, np.inf)
        x = r * np.cos(el_g) * np.cos(az_g)
        y = r * np.cos(el_g) * np.sin(az_g)
        z = r * np.sin(el_g)
        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
        return pts, valid
