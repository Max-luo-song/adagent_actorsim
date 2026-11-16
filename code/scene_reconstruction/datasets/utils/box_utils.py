import cv2
import numpy as np

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]

    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def inbbox_points(points, corner3d):
    min_xyz = corner3d[0]
    max_xyz = corner3d[-1]
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1),
        np.all(points <= max_xyz, axis=-1)
    )

# 6 个面的角点索引 (按四边形顺序)
_FACES = [
    [0, 1, 3, 2],  # bottom (min_x face)
    [4, 5, 7, 6],  # top (max_x face)
    [0, 1, 5, 4],  # front (min_y face)
    [2, 3, 7, 6],  # back (max_y face)
    [0, 2, 6, 4],  # left (min_z face)
    [1, 3, 7, 5],  # right (max_z face)
]

_NEAR = 0.01  # 近平面距离 (米)


def _project_point(pt, K):
    """3D 点 → 2D 像素 (针孔模型), pt 为 (3,) 或 (N,3)"""
    uv = pt[..., :2] / pt[..., 2:3]
    return uv @ K[:2, :2].T + K[:2, 2]


def _clip_face_to_near(corners_3d):
    """对单个面的 4 个角点 (按边顺序排列) 做近平面裁剪,
    返回裁剪后的 2D 角点列表 (可能少于或多于 4 个)."""
    # 先转世界系 corner → 相机系 corner (z=前)
    pts = corners_3d.copy()
    z = pts[:, 2]
    front = z > _NEAR

    if not front.any():
        return None  # 整个面在近平面后方

    clipped = []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        zi, zj = z[i], z[j]

        if zi > _NEAR:
            clipped.append(pts[i])
        if (zi > _NEAR) != (zj > _NEAR):
            # 边跨近平面 → 插值交点
            t = (_NEAR - zi) / (zj - zi)
            t = np.clip(t, 0.0, 1.0)
            p_clip = pts[i] + t * (pts[j] - pts[i])
            clipped.append(p_clip)

    return np.array(clipped) if clipped else None


def get_bound_2d_mask(corners_3d, K, pose, H, W):
    """生成 3D 框的 2D mask (近平面裁剪版)。

    Args:
        corners_3d: (8, 3) 局部坐标系下的 8 个角点
        K: (3, 3) 相机内参矩阵
        pose: (4, 4) cam2ego 的逆 (ego2cam)
        H, W: 图像高宽
    """
    # 变换到相机坐标系: corners_cam = corners_3d @ R^T + t^T
    corners_cam = corners_3d @ pose[:3, :3].T + pose[:3, 3:].T

    mask = np.zeros((H, W), dtype=np.uint8)

    for face_indices in _FACES:
        face_corners = corners_cam[face_indices]  # (4, 3)
        clipped = _clip_face_to_near(face_corners)
        if clipped is None or len(clipped) < 3:
            continue

        # 投影到 2D
        uv = _project_point(clipped, K)
        uv = np.round(uv).astype(np.int32)
        cv2.fillPoly(mask, [uv], 1)

    return mask


def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [
                ((0, k, l), (1, k, l)),
                ((k, 0, l), (k, 1, l)),
                ((k, l, 0), (k, l, 1)),
            ]:
                cv2.line(
                    img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness
                )

    # Draw a cross on the front face to identify front & back.
    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)


