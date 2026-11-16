"""Pure helper functions shared by render.py and the baking module.

Trajectory/geometry math, PCD/PLY I/O, and logging setup with no dependency
on Scene/GaussianModel/dataset render-pipeline state.
"""

import os
import logging
from typing import List, Dict, NamedTuple, Optional

import numpy as np
import torch
from plyfile import PlyData
from scipy.interpolate import CubicSpline, interp1d

from utils.lidar_utils import find_closest_labels

logger = logging.getLogger("render")


ALIGN_SPIN_GRID_TO_ORIGINAL_SCAN = True
SPIN_GRID_REORDER_ROWS_TO_INTRINSIC = True
SPIN_GRID_FLIP_COLUMNS = True
SPIN_GRID_COLUMN_ROLL = 0

CHERY_ELEVATION_DEG = np.array(
    [
        6.8984375, 6.5, 6.09765625, 5.6953125, 5.27734375, 4.86328125,
        4.45703125, 4.05859375, 3.6328125, 3.234375, 2.828125, 2.41796875,
        2.21484375, 2.01953125, 1.81640625, 1.60546875, 1.42578125,
        1.21484375, 1.01953125, 0.80859375, 0.62109375, 0.41015625,
        0.21484375, 0.00390625, -0.203125, -0.390625, -0.609375,
        -0.79296875, -1.01171875, -1.19921875, -1.41015625, -1.59765625,
        -1.796875, -2, -2.1953125, -2.40625, -2.6015625, -2.80859375,
        -3.00390625, -3.2265625, -3.4609375, -3.6484375, -3.8671875,
        -4.0546875, -4.27734375, -4.45703125, -4.6875, -4.8671875,
        -5.28125, -5.69921875, -6.1015625, -6.50390625, -6.91015625,
        -7.31640625, -7.71875, -8.1171875, -8.49609375, -8.8984375,
        -9.3046875, -9.71875, -10.26171875, -11.0859375, -12.0859375,
        -13.31640625, 2.31640625, 2.1171875, 1.91796875, 1.7109375,
        1.515625, 1.3203125, 1.1171875, 0.9140625, 0.71484375, 0.515625,
        0.3125, 0.109375, -0.1015625, -0.296875, -0.5, -0.69921875,
        -0.90234375, -1.10546875, -1.3046875, -1.50390625, -1.69921875,
        -1.8984375, -2.09765625, -2.30078125, -2.50390625, -2.70703125,
        -2.90625, -3.11328125, -3.34375, -3.5546875, -3.7578125,
        -3.9609375, -4.16796875, -4.3671875, -4.57421875, -4.77734375,
    ],
    dtype=np.float32,
)

CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW = np.argsort(CHERY_ELEVATION_DEG)[::-1]


def align_spin_grid_to_original_scan(spin_grid: np.ndarray) -> np.ndarray:
    """qcraft spin grid 格式转换。"""
    grid = np.asarray(spin_grid, dtype=np.float32)
    if not ALIGN_SPIN_GRID_TO_ORIGINAL_SCAN:
        return grid
    if SPIN_GRID_REORDER_ROWS_TO_INTRINSIC:
        if grid.shape[0] != CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW.shape[0]:
            # 这套行列重排/翻转是 120° chery（96 束 CHERY_ELEVATION_DEG）专属的
            # 经验对齐参数（列翻转方向是拿真实 Spin 硬件数据实测比对出来的，非
            # 推导得到）。360°（chery_lidar_360/AdaptiveDataloader，128 束）
            # 没有实测依据，直接报错，避免未经验证地套用。
            raise ValueError(
                "spin_grid beam count does not match Chery intrinsic elevations: "
                f"{grid.shape[0]} vs {CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW.shape[0]}"
            )
        reordered = np.empty_like(grid)
        reordered[CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW] = grid
        grid = reordered

    if SPIN_GRID_FLIP_COLUMNS:
        grid = np.flip(grid, axis=1)
    if SPIN_GRID_COLUMN_ROLL:
        grid = np.roll(grid, SPIN_GRID_COLUMN_ROLL, axis=1)
    return np.ascontiguousarray(grid)


def save_spin_grid_pcd_binary(file_path: str, spin_grid: np.ndarray):
    """
    Save a LiDAR spin grid to PCD binary.

    Args:
        file_path: destination .pcd file path
        spin_grid: numpy array with shape (H, W, 2), where the last dimension is
            range and intensity.
    """
    grid = np.asarray(spin_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError("spin_grid must be a (H, W, 2) array with range,intensity channels")

    h, w, _ = grid.shape
    n = h * w
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS range intensity\n"
        "SIZE 4 4\n"
        "TYPE F F\n"
        "COUNT 1 1\n"
        f"WIDTH {w}\n"
        f"HEIGHT {h}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(grid).tobytes())


def inject_into_pano(
    depth_pano: np.ndarray,
    intens_pano: np.ndarray,
    world_xyz: np.ndarray,
    world_to_cam: np.ndarray,
    beam_inclinations: np.ndarray,
    W: int,
    lidar_hfov: float,
    max_depth: float,
    intensity_val: float = 1.0,
    sample_ratio: float = 1.0,
) -> int:
    """
    将 PLY 世界坐标点注入深度全景图（in-place），返回实际影响的像素数。

    在全景图层操作而非 3D 点云层，因此：
      - 遮挡自动正确：插入物体深度 < 背景深度 → 覆盖该 beam slot
      - 背后扫描缺失自动正确：被物体占据的 beam×azimuth 格子不再有背景深度
      - 后续 pano_to_lidar_with_intensities 产生的点云天然包含两种效果
    """
    if len(world_xyz) == 0:
        return 0

    H = depth_pano.shape[0]
    M = len(world_xyz)

    if sample_ratio < 1.0:
        n_keep = max(1, int(M * sample_ratio))
        idx_s = np.random.choice(M, n_keep, replace=False)
        world_xyz = world_xyz[idx_s]
        M = len(world_xyz)

    hom = np.concatenate([world_xyz, np.ones((M, 1), dtype=np.float32)], axis=1)
    local_xyz = (hom @ world_to_cam.T)[:, :3]

    x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    dists = np.linalg.norm(local_xyz, axis=1).astype(np.float64)

    valid = dists < max_depth
    if not np.any(valid):
        return 0
    x, y, z, dists = x[valid], y[valid], z[valid], dists[valid]

    beta = lidar_hfov / 2 - np.arctan2(y, x)
    c = np.rint(beta / (lidar_hfov / W)).astype(np.int32)

    alpha = np.arctan2(z, np.sqrt(x ** 2 + y ** 2 + z ** 2))
    beam_idx = find_closest_labels(beam_inclinations, alpha)
    r = H - 1 - beam_idx

    in_bounds = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, dists = r[in_bounds], c[in_bounds], dists[in_bounds]
    if len(r) == 0:
        return 0

    flat = r * W + c
    min_ply = np.full(H * W, np.inf, dtype=np.float64)
    np.minimum.at(min_ply, flat, dists)

    has_ply = np.isfinite(min_ply)
    depth_flat = depth_pano.reshape(-1)
    intens_flat = intens_pano.reshape(-1)
    update = has_ply & ((depth_flat == 0.0) | (min_ply < depth_flat))

    depth_flat[update] = min_ply[update].astype(np.float32)
    intens_flat[update] = intensity_val

    return int(update.sum())


def compute_ego_trajectory(l2ws) -> np.ndarray:
    """
    从 gt_dynamic_model.l2ws（sensor_lidar2world 矩阵列表）中提取自车世界坐标轨迹。

    坐标系：lidargs 世界坐标系 = inv(lidar_to_world_start) @ lidar_to_world_current，
    即以第一帧 baselidar 位姿为原点、轴向与第一帧车辆前/左/上方向对齐的相对世界系。

    Args:
        l2ws: 每帧的 sensor_lidar2world (sl2w) 矩阵列表，形状 (N,) of (4,4) numpy arrays。

    Returns:
        (N, 3) numpy 数组，每行为对应帧传感器 LiDAR 在世界坐标系中的位置。
    """
    return np.array([m[:3, 3] for m in l2ws], dtype=np.float64)


def compute_cam_trajectory_from_source(
    scene_l2ws: List[np.ndarray],
    source_path: str,
    front_cam_id: int = 0,
) -> Optional[np.ndarray]:
    """
    从外参文件和 LiDAR poses 计算前置相机世界轨迹，用于与图像侧编辑对齐障碍物摆放参考点。

    Returns:
        (N, 3) 相机世界坐标轨迹，若外参文件不存在则返回 None（调用方回退到 LiDAR 轨迹）。
    """
    extr_path = os.path.join(source_path, "extrinsics", f"{front_cam_id}.txt")
    if not os.path.exists(extr_path):
        logger.warning(
            f"[cam_trajectory] 未找到前置相机外参 {extr_path}，"
            "回退到 LiDAR 轨迹（障碍物摆放参考点将与相机编辑存在偏差）"
        )
        return None
    cam_to_lidar = np.loadtxt(extr_path).astype(np.float64).reshape(4, 4)
    positions = []
    for l2w in scene_l2ws:
        cam_to_world = np.asarray(l2w, dtype=np.float64) @ cam_to_lidar
        positions.append(cam_to_world[:3, 3])
    return np.array(positions, dtype=np.float64)


def load_relative_lidar_poses(source_path: str, frame_ids: List[int]) -> Dict[int, np.ndarray]:
    """加载每帧 lidar pose 并归一化到第一帧（frame_ids[0] 为原点）。"""
    pose_dir = os.path.join(source_path, "lidar_pose")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"lidar_pose 目录不存在: {pose_dir}")
    if not frame_ids:
        raise ValueError("frame_ids 不能为空")

    ordered_frame_ids = [int(frame_id) for frame_id in frame_ids]
    base_pose_path = os.path.join(pose_dir, f"{ordered_frame_ids[0]:06d}.txt")
    base_pose = np.loadtxt(base_pose_path).reshape(4, 4)

    relative_poses = {}
    inv_base_pose = np.linalg.inv(base_pose)
    for frame_id in ordered_frame_ids:
        pose_path = os.path.join(pose_dir, f"{int(frame_id):06d}.txt")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"缺少 lidar pose: {pose_path}")
        current_pose = np.loadtxt(pose_path).reshape(4, 4)
        relative_poses[int(frame_id)] = inv_base_pose @ current_pose
    return relative_poses


def normalize_scene_range(window, num_frames: int):
    """Normalize [start, end, step] against the ordered scene frame list."""
    assert num_frames >= 0
    if not isinstance(window, (list, tuple)) or len(window) == 0:
        return 0, num_frames, 1

    start = int(window[0])
    end = int(window[1]) if len(window) > 1 else num_frames
    step = int(window[2]) if len(window) > 2 else 1

    if end == -1:
        end = num_frames

    start = max(0, min(start, num_frames))
    end = max(start, min(end, num_frames))
    step = max(step, 1)
    return start, end, step


def sample_scene_indices_by_distance(
    ego_trajectory: np.ndarray,
    start_idx: int,
    end_idx: int,
    spacing_m: float,
) -> list:
    """Sample trajectory positions by accumulated travel distance within [start_idx, end_idx)."""
    if end_idx <= start_idx:
        return []
    if spacing_m <= 0:
        return list(range(start_idx, end_idx))

    sampled = [start_idx]
    accumulated = 0.0
    for scene_idx in range(start_idx + 1, end_idx):
        accumulated += float(
            np.linalg.norm(ego_trajectory[scene_idx] - ego_trajectory[scene_idx - 1])
        )
        if accumulated + 1e-6 >= spacing_m:
            sampled.append(scene_idx)
            accumulated = 0.0
    return sampled


def interp_waypoints(waypoints: list, method: str, num_frames: int) -> np.ndarray:
    """将 waypoints 插值为 num_frames 帧的 [dx,dy,dz] 序列。

    method: 'linear' | 'cubic'（cubic 需要 >=3 个 waypoints，否则降级为 linear）
    """
    wp = np.array(waypoints, dtype=np.float64)  # (K, 3)
    K = len(wp)
    if K == 1 or num_frames == 1:
        return np.tile(wp[0], (num_frames, 1))
    t_src = np.linspace(0.0, 1.0, K)
    t_dst = np.linspace(0.0, 1.0, num_frames)
    result = np.zeros((num_frames, 3), dtype=np.float64)
    for dim in range(3):
        if method == "cubic" and K >= 3:
            f = CubicSpline(t_src, wp[:, dim])
        else:
            f = interp1d(t_src, wp[:, dim], kind="linear", fill_value="extrapolate")
        result[:, dim] = f(t_dst)
    return result


class EditObjInfo(NamedTuple):
    delete_obj_ids: list
    obj_id_offset_pairs: list
    add_id_path_pairs: list
    novel_poses: list


def get_logger(path, log_path):
    import logging
    logger = logging.getLogger("render")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = os.path.join(log_path, "outputs.log")
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in logger.handlers
    )
    if not has_file:
        fileinfo = logging.FileHandler(log_file)
        fileinfo.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        logger.addHandler(fileinfo)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        controlshow = logging.StreamHandler()
        controlshow.setLevel(logging.INFO)
        controlshow.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
        logger.addHandler(controlshow)

    return logger


def save_pcd_binary(file_path: str, points: np.ndarray):
    """
    Save Nx>=4 array to PCD binary with fields x y z intensity.

    Args:
        file_path: destination .pcd file path
        points: numpy array shape (N, >=4) where columns are x,y,z,intensity,...
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError("points must be a (N, >=4) array with x,y,z,intensity columns")

    N = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        data = pts[:, :4].astype(np.float32)
        f.write(data.tobytes())


def build_world_transform_tensors(model_to_world: np.ndarray):
    from scipy.spatial.transform import Rotation
    model_world_rotation = torch.tensor(
        model_to_world[:3, :3], dtype=torch.float32, device="cuda"
    )
    model_world_translation = torch.tensor(
        model_to_world[:3, 3], dtype=torch.float32, device="cuda"
    )
    model_world_quaternion_np = Rotation.from_matrix(model_to_world[:3, :3]).as_quat()
    model_world_quaternion = torch.tensor(
        [
            model_world_quaternion_np[3],
            model_world_quaternion_np[0],
            model_world_quaternion_np[1],
            model_world_quaternion_np[2],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    return model_world_rotation, model_world_translation, model_world_quaternion


def read_single_ply_to_obj(
    ply_file_path: str,
    obj_class: str = None,
    sim_pose: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
    intensity: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    读取单个 PLY 文件，转换为单个 obj 字典
    """
    try:
        ply_data = PlyData.read(ply_file_path)
        vertices = ply_data["vertex"]
        num_vertices = len(vertices)
    except Exception as e:
        return None

    xyz = torch.tensor(
        np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1),
        dtype=torch.float32,
        device="cuda",
    )
    if scale_factor != 1.0:
        xyz = xyz * scale_factor  # xyz 也随 scale_factor 放大，保持与 scaling 一致

    opacity = torch.tensor(
        vertices["opacity"], dtype=torch.float32, device="cuda"
    ).unsqueeze(1)
    opacity = torch.clamp(opacity, 0.0, 1.0)
    opacity[:] = 1.0  # 不透明度全设为 1

    scaling = torch.tensor(
        np.stack(
            [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1
        ),
        dtype=torch.float32,
        device="cuda",
    )
    scaling = torch.exp(scaling) * scale_factor

    rot = torch.tensor(
        np.stack(
            [
                vertices["rot_0"],
                vertices["rot_1"],
                vertices["rot_2"],
                vertices["rot_3"],
            ],
            axis=1,
        ),
        dtype=torch.float32,
        device="cuda",
    )

    # LiDAR color: [intensity, raydrop]，intensity 默认与保存模型时的值对齐
    color = torch.ones((num_vertices, 2), dtype=torch.float32, device="cuda")
    color[:, 0] = intensity
    color[:, 1] = 1.0  # raydrop

    if sim_pose is not None:
        if isinstance(sim_pose, np.ndarray) or isinstance(sim_pose, list):
            sim_pose = torch.tensor(
                sim_pose, dtype=torch.float32, device="cuda"
            ).reshape(4, 4)
        else:
            sim_pose = sim_pose.to(device="cuda", dtype=torch.float32).reshape(4, 4)

    if sim_pose is not None:
        sim_pose = torch.tensor(sim_pose, dtype=torch.float32).reshape(4, 4)

    single_obj = {
        "obj_class": obj_class,
        "xyz": xyz,
        "color": color,
        "opacity": opacity,
        "scaling": scaling,
        "rot": rot,
        "sim_pose": sim_pose,
    }

    return single_obj


def batch_read_plys_to_insert_objs(
    edit_obj_info: EditObjInfo,
) -> List[Dict[str, torch.Tensor]]:
    """
    批量读取多个 PLY 文件，生成 insert_objs 列表（一个文件对应一个 obj）

    Args:
        edit_obj_info: 包含 add_id_path_pairs 的类实例（存储 (obj_class, 文件路径) 对）

    Returns:
        insert_objs: 列表，每个元素是一个 obj 字典
    """
    insert_objs = []
    for obj_class, sim_pose, obj_pcd_filepath in edit_obj_info.add_id_path_pairs:
        single_obj = read_single_ply_to_obj(obj_pcd_filepath, obj_class, sim_pose)
        if single_obj is not None:
            insert_objs.append(single_obj)
    return insert_objs
