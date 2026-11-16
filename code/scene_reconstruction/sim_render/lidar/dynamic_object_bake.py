"""雷达可控编辑：动态障碍物插入，保存为标准 scaffold-gaussian model，写入 checkpoint 文件。

在 scaffold-gaussian model 中，opacity/color/scale 等 3DGS 属性，基于 MLP 网络
根据视线向量、距离进一步映射得到，这几个 MLP（mlp_opacity/mlp_cov/mlp_color/
mlp_raydrop）是拿原场景的真实雷达数据训练出来的。对于插入的新动态障碍物，没有
对应的真实数据，这些属性无法通过原场景训练好的 MLP 网络进行映射。

因此这里将新增动态障碍物 gaussian model 中对应的 mlp_opacity/mlp_cov/
mlp_color/mlp_raydrop 权重设置为固定值，使其输出不再依赖视线向量和距离，只
取决于我们写入的 anchor_feat 数据。保存后的 checkpoint 与标准渲染管线兼容，
任何加载该 model_path 的渲染流程（含在线 Renderer）都能直接识别这个动态障碍物。

每个 anchor 可以展开出 n_offsets 个候选高斯点，这里只手工构造了第 0 个候选点
的数据，因此权重设置还需要让 opacity 只在第 0 个候选点上为正（其余为负、会被
渲染管线丢弃）。
"""

import os
import json
import shutil
import logging
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData

from arguments import ModelParams
from scene.gaussian_model import GaussianModel

from sim_render.lidar.render_utils import interp_waypoints

logger = logging.getLogger("render")


# 权重固定为该常量时，配合 Tanh/Sigmoid 激活函数，输出会饱和到接近 ±1（或
# 0/1），基本不再随输入变化。
FIXED_WEIGHT_MAGNITUDE = 10.0


def make_passthrough_opacity_mlp(mlp: torch.nn.Sequential, n_offsets: int) -> None:
    """mlp_opacity 结构为 Linear→ReLU→Linear→Tanh（见 GaussianModel 定义）。
    把两个 Linear 层的权重清零、只保留 bias，使输出不再依赖输入：第一层 bias
    为正，经 ReLU 后仍为正；第二层 bias 第 0 位为 +FIXED_WEIGHT_MAGNITUDE、
    其余位为负，经 Tanh 饱和后，第 0 个候选点的透明度输出接近 +1（可见），
    其余候选点接近 -1（渲染管线视为不可见，会被丢弃）。"""
    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].bias.fill_(FIXED_WEIGHT_MAGNITUDE)
        mlp[2].weight.zero_()
        bias2 = torch.full((n_offsets,), -FIXED_WEIGHT_MAGNITUDE, device=mlp[2].bias.device)
        bias2[0] = FIXED_WEIGHT_MAGNITUDE
        mlp[2].bias.copy_(bias2)


def make_passthrough_cov_mlp(mlp: torch.nn.Sequential, feat_dim: int) -> None:
    """mlp_cov 结构为 Linear→ReLU→Linear（无激活函数），输出 6 维（每个候选点），
    渲染时外部代码对前 2 维取 sigmoid 得到缩放系数、对后 4 维归一化得到旋转
    四元数（见 gaussian_renderer/__init__.py 的 generate_neural_gaussians）。

    这里把权重固定为：第 0 个候选点的前 2 维恒为 0（sigmoid(0)=0.5，配合
    create_synthetic_dynamic_object 里已经把目标大小的 2 倍取了对数存进
    _scaling，两者相乘后精确等于目标大小），后 4 维直接原样输出我们写进
    anchor_feat[0:4] 的旋转四元数。
    """
    with torch.no_grad():
        mlp[0].weight.zero_()
        for i in range(4):
            mlp[0].weight[i, i] = 1.0
        mlp[0].bias.zero_()
        mlp[0].bias[:4] = FIXED_WEIGHT_MAGNITUDE
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        for i in range(4):
            mlp[2].weight[2 + i, i] = 1.0
        mlp[2].bias[2:6] = -FIXED_WEIGHT_MAGNITUDE


def make_passthrough_color_mlp(mlp: torch.nn.Sequential, feat_dim: int) -> None:
    """mlp_color 结构为 Linear→ReLU→Linear→Sigmoid。把权重固定为：第 0 个
    候选点的反射强度直接原样输出我们写进 anchor_feat[4] 的值（sigmoid 前的
    logit，见 create_synthetic_dynamic_object 里的 intensity_logit）。"""
    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].weight[4, 4] = 1.0
        mlp[0].bias.zero_()
        mlp[0].bias[4] = FIXED_WEIGHT_MAGNITUDE
        mlp[2].weight.zero_()
        mlp[2].bias.zero_()
        mlp[2].weight[0, 4] = 1.0
        mlp[2].bias[0] = -FIXED_WEIGHT_MAGNITUDE


def make_passthrough_raydrop_mlp(mlp: torch.nn.Sequential, n_offsets: int) -> None:
    """mlp_raydrop 结构为 Linear→ReLU→Linear→Sigmoid。把权重固定为：第 0 个
    候选点恒输出接近 1（渲染管线按 raydrop>0.5 判定为有效点，不丢弃）。

    这里 bias 必须是正数，不能是负数：之前写成负数时，插入的车被渲染管线
    整体当成无效点丢弃，视频里车辆全部消失过一次。
    """
    with torch.no_grad():
        mlp[0].weight.zero_()
        mlp[0].bias.fill_(FIXED_WEIGHT_MAGNITUDE)
        mlp[2].weight.zero_()
        bias2 = torch.zeros(n_offsets, device=mlp[2].bias.device)
        bias2[0] = FIXED_WEIGHT_MAGNITUDE
        mlp[2].bias.copy_(bias2)


def read_vanilla_3dgs_ply(ply_path: str):
    """读取 vanilla 3DGS 格式的 PLY（x,y,z / opacity / scale_0-2 / rot_0-3），
    与雷达侧 anchor+MLP 格式不同，这里只提取造型需要的字段（不含 SH 颜色，雷达
    侧用固定反射强度代替，见模块顶部说明）。返回 (xyz, scale_3d, rot) 均为
    (N, ...) numpy 数组。"""
    ply_data = PlyData.read(ply_path)
    vertices = ply_data["vertex"]
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    scale_3d = np.exp(
        np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1)
    ).astype(np.float32)
    rot = np.stack(
        [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1
    ).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-8)
    return xyz, scale_3d, rot


def create_synthetic_dynamic_object(
    ply_path: str,
    dataset: ModelParams,
    target_intensity: float = 0.2,
) -> "GaussianModel":
    """把外部 PLY 资产转换成一个可被标准渲染管线加载的 GaussianModel（手工填入
    anchor 数据 + 固定 MLP 权重，见模块顶部说明）。返回的模型可直接调用
    save_ply/save_mlp_checkpoints(mode="unite") 存成标准 checkpoint 目录。
    """
    xyz, scale_3d, rot = read_vanilla_3dgs_ply(ply_path)
    n = xyz.shape[0]
    device = "cuda"

    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        dataset.color_channel,
    )
    n_offsets = dataset.n_offsets
    feat_dim = dataset.feat_dim

    # anchor 位置，直接用资产自身的点位置（局部坐标系；世界坐标由渲染时的
    # instances_info.json 逐帧位姿决定，这里不做任何平移）
    anchor = torch.tensor(xyz, dtype=torch.float32, device=device)

    # 每个 anchor 对应 n_offsets 个候选点，这里只填了第 0 个候选点需要的形状，
    # 其余值不参与渲染（上面固定权重已让它们的 opacity 恒为负，被渲染管线丢弃）
    offset = torch.zeros(n, 3, n_offsets, dtype=torch.float32, device=device)

    # 最终 scale = exp(_scaling[:,3:5]) * sigmoid(mlp_cov 输出[:2])，mlp_cov
    # 的缩放输出已固定为 0（sigmoid(0)=0.5），所以这里用 log(2*目标大小) 精确
    # 抵消这个 0.5，使最终 scale 等于目标大小。雷达渲染只有 2 个自由度，用
    # 资产的 scale_0/scale_1 近似。
    scale_2d = scale_3d[:, :2]
    scaling = torch.zeros(n, 5, dtype=torch.float32, device=device)
    scaling[:, 3:5] = torch.tensor(np.log(2.0 * np.clip(scale_2d, 1e-4, None)), device=device)
    scaling[:, 0:3] = torch.tensor(np.log(np.clip(scale_3d, 1e-4, None)), device=device)  # 占位，不影响最终渲染

    # anchor_feat：0:4 是旋转四元数（mlp_cov 会直接原样输出这四个数），4 是
    # 反射强度的 logit（mlp_color 会直接原样输出这个数），其余不用，置 0
    anchor_feat = torch.zeros(n, feat_dim, dtype=torch.float32, device=device)
    anchor_feat[:, 0:4] = torch.tensor(rot, dtype=torch.float32, device=device)
    intensity_logit = float(np.log(target_intensity / (1.0 - target_intensity)))
    anchor_feat[:, 4] = intensity_logit

    # opacity/rotation：渲染路径完全不读这两个张量（真正决定渲染结果的是上面的
    # anchor_feat + MLP），占位即可，保持形状正确避免 save_ply 报错
    opacity = torch.ones(n, 1, dtype=torch.float32, device=device)
    rotation = torch.zeros(n, 4, dtype=torch.float32, device=device)
    rotation[:, 0] = 1.0

    gaussians._anchor = torch.nn.Parameter(anchor.requires_grad_(False))
    gaussians._offset = torch.nn.Parameter(offset.requires_grad_(False))
    gaussians._anchor_feat = torch.nn.Parameter(anchor_feat.requires_grad_(False))
    gaussians._scaling = torch.nn.Parameter(scaling.requires_grad_(False))
    gaussians._rotation = torch.nn.Parameter(rotation.requires_grad_(False))
    gaussians._opacity = torch.nn.Parameter(opacity.requires_grad_(False))

    make_passthrough_opacity_mlp(gaussians.mlp_opacity, n_offsets)
    make_passthrough_cov_mlp(gaussians.mlp_cov, feat_dim)
    make_passthrough_color_mlp(gaussians.mlp_color, feat_dim)
    make_passthrough_raydrop_mlp(gaussians.mlp_raydrop, n_offsets)

    logger.info(
        f"[synthetic_dynamic_object] {ply_path} → {n} anchors, "
        f"target_intensity={target_intensity}（固定值，非资产真实反射率，见函数说明）"
    )
    return gaussians


def compute_rigid_add_world_poses(
    waypoints: list,
    method: str,
    time_window,
    ego_trajectory_local: np.ndarray,
    edit_frame_ids: list,
    lidar_to_world_start: np.ndarray,
    asset_yaw_offset: float = 0.0,
) -> dict:
    """计算 RigidNodes.add.waypoints 每一帧的绝对世界坐标系 4x4 位姿（含朝向），
    用于写进 instances_info.json（格式与真实动态物体标注一致，见
    Chery_Dataloader.load_dynamic_obj_info）。朝向算法与图像侧
    rigid.py._apply_trajectory_and_rotation 对齐：车头朝向由相邻帧位置差分
    算出的行驶方向决定（静止帧、首尾帧的特殊处理见下方实现）。

    Args:
        asset_yaw_offset: 资产建模朝向与行驶方向的偏差修正量（弧度）。不同资产
            的建模朝向约定不同，默认值 0 对应"资产朝向已经与行驶方向一致"。

    Returns:
        {frame_id: (4,4) np.ndarray}，只包含 time_window 内的帧。
    """
    N = len(ego_trajectory_local)
    offsets_full = interp_waypoints(waypoints, method, N)  # (N, 3) 局部系
    positions_local = ego_trajectory_local + offsets_full  # (N, 3)

    dirs = np.zeros_like(positions_local)
    if N > 1:
        dirs[:-1] = positions_local[1:] - positions_local[:-1]
        dirs[-1] = dirs[-2]
    speed_eps = 1e-4
    for i in range(1, N):
        if np.linalg.norm(dirs[i, :2]) < speed_eps:
            dirs[i] = dirs[i - 1]
    if N > 0 and np.linalg.norm(dirs[0, :2]) < speed_eps:
        dirs[0] = np.array([1.0, 0.0, 0.0])
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    yaw = np.arctan2(dirs[:, 1], dirs[:, 0]) + asset_yaw_offset
    for i in range(1, N):
        delta = yaw[i] - yaw[i - 1]
        if delta > np.pi:
            yaw[i] -= 2 * np.pi
        elif delta < -np.pi:
            yaw[i] += 2 * np.pi

    if time_window is not None:
        start_idx = max(0, int(time_window[0]))
        end_idx = min(N, int(time_window[1]) + 1)
    else:
        start_idx, end_idx = 0, N

    poses_world = {}
    for frame_idx in range(start_idx, end_idx):
        yaw_i = float(yaw[frame_idx])
        rot_z = np.array(
            [
                [np.cos(yaw_i), -np.sin(yaw_i), 0.0],
                [np.sin(yaw_i), np.cos(yaw_i), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pose_local = np.eye(4, dtype=np.float64)
        pose_local[:3, :3] = rot_z
        pose_local[:3, 3] = positions_local[frame_idx].astype(np.float64)
        pose_world = lidar_to_world_start.astype(np.float64) @ pose_local
        frame_id = int(edit_frame_ids[frame_idx])
        poses_world[frame_id] = pose_world.astype(np.float32)
    return poses_world


def write_placeholder_dynamic_pcd(
    gt_data_root: str, frame_id: int, instance_id: int, xyz_local: np.ndarray
) -> None:
    """从资产的真实点位置里抽样，写一份 (x,y,z,intensity) 格式的 npy 文件到
    Chery_Dataloader 约定的动态物体点云目录（lidar_preprocessed_data/dynamic_pcd/）。

    这份 npy 的作用仅仅是让 Chery_Dataloader.load_dynamic_pcd 的"至少 20 点"
    有效性校验通过，本身不参与渲染——渲染读取的是 point_cloud.ply（标准
    checkpoint），不会再读这份 npy 的具体内容。每个 block 只要有一帧写了这个
    文件就能通过校验，不需要逐帧都写。"""
    n = min(len(xyz_local), 200)
    idx = np.random.choice(len(xyz_local), n, replace=False) if len(xyz_local) > n else np.arange(len(xyz_local))
    pts = xyz_local[idx]
    intensity = np.full((n, 1), 0.45, dtype=np.float32)
    data = np.concatenate([pts.astype(np.float32), intensity], axis=1)
    out_dir = os.path.join(gt_data_root, "lidar_preprocessed_data", "dynamic_pcd", str(frame_id).zfill(3))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{str(frame_id).zfill(3)}_obj{str(instance_id).zfill(3)}.npy")
    np.save(out_path, data)


def merge_instances_info_json(
    gt_data_root: str,
    source_instances_json: str,
    instance_id: int,
    raw_id: int,
    class_name: str,
    poses_world_by_frame: dict,
    box_size: list,
) -> None:
    """把新物体的标注合并进 instances_info.json（读 source_instances_json 做
    基础，只新增一条记录，不覆盖场景里其它真实物体的标注），写到
    gt_data_root/instances/instances_info.json。"""
    data = {}
    if os.path.exists(source_instances_json):
        with open(source_instances_json) as f:
            data = json.load(f)
    else:
        logger.warning(f"[merge_instances_info] 源标注文件不存在，将新建: {source_instances_json}")

    frame_idx_sorted = sorted(poses_world_by_frame.keys())
    data[str(instance_id)] = {
        "class_name": class_name,
        "id": raw_id,
        "frame_annotations": {
            "frame_idx": frame_idx_sorted,
            "obj_to_world": [poses_world_by_frame[f].tolist() for f in frame_idx_sorted],
            "obj_to_ego": [poses_world_by_frame[f].tolist() for f in frame_idx_sorted],  # 占位，未见下游读取
            "box_size": [box_size for _ in frame_idx_sorted],
        },
    }

    out_path = os.path.join(gt_data_root, "instances", "instances_info.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.info(f"[merge_instances_info] instance_id={instance_id} → {out_path}（共 {len(data)} 个物体）")


def bake_rigid_add_as_dynamic_object(
    ply_path: str,
    waypoints: list,
    method: str,
    time_window,
    ego_trajectory_local: np.ndarray,
    edit_frame_ids: list,
    lidar_to_world_start: np.ndarray,
    dataset: ModelParams,
    gt_data_root: str,
    source_instances_json: str,
    block_info: dict,
    iteration: int,
    instance_id: int,
    class_name: str = "Vehicle",
    target_intensity: float = 0.2,
    asset_yaw_offset: float = 0.0,
) -> None:
    """把 RigidNodes.add.waypoints 插入的车辆保存成一个正式的动态物体：每个
    涉及的 block 各存一份 checkpoint 文件，instances_info.json 合并写入逐帧
    世界位姿，每个 block 补一份占位点云。

    这是与"渲染时临时叠加"（render.py 的 create_rigid_add_per_frame_objs，
    inject_direct 路径）独立的另一种方法：临时叠加不写文件，只影响当次渲染
    结果；这里则会写入 model_path，此后任何加载该 model_path 的渲染流程
    （含在线 Renderer）都能直接识别这辆车，不需要再传 edit_yaml。
    """
    model_id = instance_id + 1  # Chery_Dataloader.instance_id_to_model_id 的约定

    poses_world = compute_rigid_add_world_poses(
        waypoints, method, time_window, ego_trajectory_local, edit_frame_ids,
        lidar_to_world_start, asset_yaw_offset=asset_yaw_offset,
    )
    if not poses_world:
        logger.warning("[bake_rigid_add] time_window 内没有帧，跳过")
        return

    xyz_local, _, _ = read_vanilla_3dgs_ply(ply_path)
    box_size = [
        float(xyz_local[:, 0].max() - xyz_local[:, 0].min()),
        float(xyz_local[:, 1].max() - xyz_local[:, 1].min()),
        float(xyz_local[:, 2].max() - xyz_local[:, 2].min()),
    ]

    frame_ids_in_window = set(poses_world.keys())
    for block_id, block_frame_times in block_info.items():
        if not (set(block_frame_times) & frame_ids_in_window):
            continue  # 这个 block 完全不在插入车辆的可见窗口内，不用存

        gaussians = create_synthetic_dynamic_object(ply_path, dataset, target_intensity)
        save_dir = os.path.join(gt_data_root, str(model_id), str(block_id), f"iteration_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
        gaussians.save_mlp_checkpoints(save_dir, mode="unite")
        logger.info(f"[bake_rigid_add] model_id={model_id} block_id={block_id} → {save_dir}")

        # 每个 block 只需一帧占位点云即可通过 load_dynamic_pcd 的有效性校验
        placeholder_frame = next(iter(sorted(set(block_frame_times) & frame_ids_in_window)))
        write_placeholder_dynamic_pcd(gt_data_root, placeholder_frame, instance_id, xyz_local)

    raw_id = 900000 + instance_id  # 占位 raw_id，避开真实标注 id 空间
    merge_instances_info_json(
        gt_data_root, source_instances_json, instance_id, raw_id, class_name,
        poses_world, box_size,
    )


def save_edited_gaussian_state(
    model_id_scene_info: dict,
    save_root: str,
    block_id: int,
    iteration: int,
    edit_yaml_src: Optional[str] = None,
    cfg_args_src: Optional[str] = None,
) -> None:
    """将编辑后（delete/move 已应用）的 Gaussian 模型 PLY + checkpoints.pth 保存到 save_root。

    保存结构：
        {save_root}/{model_id}/{block_id}/iteration_{iter}/point_cloud.ply
        {save_root}/{model_id}/{block_id}/iteration_{iter}/checkpoints.pth
    """
    for model_id, gs_view in model_id_scene_info.items():
        save_dir = os.path.join(
            save_root, str(model_id), str(block_id), f"iteration_{iteration}"
        )
        os.makedirs(save_dir, exist_ok=True)
        gs_view.gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
        gs_view.gaussians.save_mlp_checkpoints(save_dir, mode="unite")
        logger.info(
            f"[save_ckpt] model_id={model_id} block_id={block_id} "
            f"iteration={iteration} → {save_dir}"
        )

    if edit_yaml_src and os.path.exists(edit_yaml_src):
        shutil.copy(edit_yaml_src, os.path.join(save_root, "edit_config.yaml"))
        logger.info(f"[save_ckpt] edit_config.yaml → {save_root}/edit_config.yaml")

    if cfg_args_src and os.path.exists(cfg_args_src):
        shutil.copy(cfg_args_src, os.path.join(save_root, "cfg_args"))
        logger.info(f"[save_ckpt] cfg_args → {save_root}/cfg_args")


def compute_final_offset_from_windows(offset_windows: list, N: int) -> list:
    """从 offset_windows 计算第 N-1 帧的累积偏移，用于保存模型时的固定平移量。

    与图像侧 edit_trajectory_interpolate 相同语义：
      offset_windows = [[start_f, end_f, [dx,dy,dz]], ...]
      窗口内 smoothstep 插值，窗口结束后偏移保持。
    """
    offset = np.zeros(3, dtype=np.float64)
    last_f = max(0, N - 1)
    for win in (offset_windows or []):
        start_f, end_f, delta = int(win[0]), int(win[1]), np.array(win[2], dtype=np.float64)
        if start_f == end_f or last_f >= end_f:
            offset += delta
        elif last_f >= start_f:
            t = (last_f - start_f) / (end_f - start_f)
            t = t * t * (3 - 2 * t)  # smoothstep
            offset += delta * t
    return offset.tolist()
