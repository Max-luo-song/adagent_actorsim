
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import os
import json
import shutil
import torch
import numpy as np
import sys
import subprocess

from typing import Optional
from scipy.spatial.transform import Rotation

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

from scene import Scene
import time
import yaml
from gaussian_renderer import renderComposite
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from scene.unet import UNet

from utils.lidar_utils import pano_to_lidar_with_intensities, filter_pcd
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2
import logging
logger = logging.getLogger("render")
logger.setLevel(logging.INFO)
# avoid propagating to root logger (prevents duplicate printing)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)

from typing import NamedTuple

from sim_render.lidar.render_utils import (
    align_spin_grid_to_original_scan,
    inject_into_pano,
    compute_ego_trajectory,
    compute_cam_trajectory_from_source,
    load_relative_lidar_poses,
    normalize_scene_range,
    sample_scene_indices_by_distance,
    interp_waypoints,
    EditObjInfo,
    get_logger,
    save_pcd_binary,
    build_world_transform_tensors,
    read_single_ply_to_obj,
    batch_read_plys_to_insert_objs,
)
from sim_render.lidar.dynamic_object_bake import (
    bake_rigid_add_as_dynamic_object,
    save_edited_gaussian_state,
    compute_final_offset_from_windows,
)


# ---------------------------------------------------------------------------
# 编辑场景专用：逐帧动态插入物体（RigidNodes.add + waypoints）
#
# 仅在本文件末尾 `if __name__ == "__main__":` 这条命令行路径使用，即直接
# 执行 `python render.py --edit_yaml ...` 时才会调用到。在线 Renderer
# （lidar_inference_online.py）走的是 prepare_render_cache/render_cached_frame，
# 不会调用这里的函数，永远不会产生 insert_objs，因此这块及 render_set 里的
# inject_direct 分支不影响线上渲染。
# ---------------------------------------------------------------------------


def create_rigid_add_per_frame_objs(
    ply_path: str,
    waypoints: list,
    method: str,
    time_window,
    ego_trajectory: np.ndarray,
    edit_frame_ids: list,
) -> list:
    """根据 waypoints（相对 ego 轨迹的偏移）创建逐帧 insert_obj。

    每个对象带 frame_id 字段，仅在对应帧被 render_set 的 inject_direct 分支使用
    （不进入 renderComposite / Gaussian 光栅化路径）。
    time_window: [start_frame, end_frame]（场景级帧号），None 表示全部帧。

    语义与【图像侧】(models/nodes/rigid.py _apply_trajectory_and_rotation) 对齐：
    waypoints 在【整段场景 N 帧】上插值决定运动轨迹，time_window 只作为
    【可见性窗口】（控制哪些帧渲染这辆车），不改变每帧位置。这样雷达侧和图像侧
    对同一份 yaml 得到相同的速度与位置。
    """
    N = len(ego_trajectory)
    if N <= 0:
        return []

    offsets_full = interp_waypoints(waypoints, method, N)  # (N, 3)

    if time_window is not None:
        start_idx = max(0, int(time_window[0]))
        end_idx = min(N, int(time_window[1]) + 1)
    else:
        start_idx, end_idx = 0, N
    if end_idx - start_idx <= 0:
        return []

    objs = []
    for frame_idx in range(start_idx, end_idx):
        ego_pos = ego_trajectory[frame_idx]
        off = offsets_full[frame_idx]

        sim_pose = np.eye(4, dtype=np.float32)
        sim_pose[0, 3] = float(ego_pos[0]) + float(off[0])
        sim_pose[1, 3] = float(ego_pos[1]) + float(off[1])
        sim_pose[2, 3] = float(ego_pos[2]) + float(off[2])

        raw_frame_id = int(edit_frame_ids[frame_idx])
        obj = read_single_ply_to_obj(ply_path, obj_class="rigid_add", sim_pose=sim_pose)
        if obj is None:
            continue
        obj["frame_id"] = raw_frame_id
        obj["inject_direct"] = True
        obj["inject_sample_ratio"] = 1.0
        objs.append(obj)

    if objs:
        logger.info(
            f"[RigidNodes.add] waypoints({method}) → {len(objs)} 逐帧 insert_obj "
            f"frames {int(edit_frame_ids[start_idx])}–{int(edit_frame_ids[end_idx - 1])}"
        )
    return objs


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera
    valid_mask: np.array
    gt_mask: torch.Tensor


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool
    model_world_rotation: "torch.Tensor" = None
    model_world_translation: "torch.Tensor" = None
    model_world_quaternion: "torch.Tensor" = None




class RenderCache(NamedTuple):
    gt_dynamic_model: object
    valid_timestamp_model: dict
    model_id_scene_info: dict
    static_views: list
    view_by_timestamp: dict
    background: torch.Tensor




def render_set(
    gt_dynamic_model,
    dataset,
    name,
    iteration,
    valid_timestamp_model,
    model_id_scene_info,
    views,
    pipeline,
    background,
    insert_objs,
):
    render_outputs={}
    path_name = dataset.model_path.split("/")
    render_path = os.path.join(dataset.output_path, "renders")
    gt_path = os.path.join(dataset.output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    t_list = []
    total_frames = len(views)

    total_render_start = time.time()
    original_l2ws = gt_dynamic_model.l2ws

    raydrop_unet_available = False
    ckpt_path = os.path.join(dataset.model_path, "ckpt", "refine.pth")
    if not os.path.exists(ckpt_path):
        logger.error(f"UNet checkpoint 不存在: {ckpt_path}")
    else:
        raydrop_unet_available = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # UNet输入通道从3→1
        unet = UNet(in_channels=1, out_channels=1)
        state = torch.load(ckpt_path, map_location="cpu")
        unet.load_state_dict(state)
        unet.to(device)
        unet.eval()

    lidar_hfov = gt_dynamic_model.get_fov_horizontal()

    # insert_objs 里每个 obj 按是否带 inject_direct=True 标记分成两类：
    # - render_objs（不带该标记）：走 renderComposite 的高斯光栅化流程，
    #   跟场景里的真实物体一样参与渲染，用于静态 add（RigidNodes.add 不带
    #   waypoints 的情况）。
    # - inject_objs（带该标记）：不进入光栅化，而是在渲染完成后直接把点
    #   投影写入深度/强度全景图（见下方 inject_into_pano 调用），用于
    #   waypoints 产生的逐帧插入物体，按 frame_id 只在对应帧生效。
    if insert_objs:
        render_objs = [o for o in insert_objs if not o.get("inject_direct", False)]
        inject_objs = [o for o in insert_objs if o.get("inject_direct", False)]
    else:
        render_objs = insert_objs
        inject_objs = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_timestamp = view.image_name
        cur_frame_id = int(render_timestamp)
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            (
                model_world_rotation,
                model_world_translation,
                model_world_quaternion,
            ) = build_world_transform_tensors(model_to_world)
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False,
                    model_world_rotation=model_world_rotation,
                    model_world_translation=model_world_translation,
                    model_world_quaternion=model_world_quaternion,
                )
            )

        torch.cuda.synchronize()
        t0 = time.time()
        render_pkg = renderComposite(
            view,
            background,
            pipeline,
            valid_model_info,
            dataset.max_depth,
            insert_objs=render_objs,
            retain_grad=False,
        )
        torch.cuda.synchronize()
        t1 = time.time()
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info(
            f"[GPU] 当前帧显存占用: allocated={mem_allocated:.2f} MB, reserved={mem_reserved:.2f} MB"
        )
        t_list.append(t1 - t0)

        rendering = render_pkg["render"]
        render_intensity = rendering[0:1, ...]
        depth = render_pkg["depth"]

        gt = view.original_image.cuda()
        ray_drop = gt[0:1, ...]
        gt_intensity = (gt[1:2, ...] * ray_drop).detach().cpu().numpy()
        gt_depth = (gt[2:3, ...] * ray_drop).detach().cpu().numpy()
        render_raydrop = rendering[1:2, ...]

        if raydrop_unet_available:
            raydrop_refine = unet(render_raydrop)
            render_raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
        else:
            render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)

        render_intensity = render_intensity * render_raydrop_mask
        depth = depth * render_raydrop_mask

        if True:
            depth_distortion_aware = render_pkg["mid_depth_diff"]
            depth_distortion_aware = torch.where(depth_distortion_aware < 0.3, 1, 0)
            render_intensity = render_intensity * depth_distortion_aware
            depth = depth * depth_distortion_aware

        depth_numpy = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()

        # 在 pano_to_lidar_with_intensities 之前直接改 depth/intensity 全景图，
        # 遮挡与背向缺失自动正确；inject_objs 为空时不执行，不影响原有结果。
        frame_inject_objs = [
            o for o in inject_objs
            if o.get("frame_id") is None or o["frame_id"] == cur_frame_id
        ]
        if frame_inject_objs:
            depth_pano = depth_numpy[0].copy()
            intens_pano = intensity_numpy[0].copy()
            beam_inclinations = view.beam_inclinations.detach().cpu().numpy()
            total_px = 0
            for each_obj in frame_inject_objs:
                if each_obj.get("sim_pose") is None:
                    continue
                obj_xyz = each_obj["xyz"]
                M_obj = obj_xyz.shape[0]
                xyz_hom = torch.cat(
                    [obj_xyz, torch.ones(M_obj, 1, device=obj_xyz.device, dtype=obj_xyz.dtype)],
                    dim=1,
                )
                # read_single_ply_to_obj 返回的 sim_pose 不保证在 obj_xyz 所在 device
                # 上（历史实现里最后会被重新包一层 CPU tensor），显式 as_tensor 到目标
                # device 以避免 cuda/cpu 张量相乘崩溃。
                sim_pose_t = torch.as_tensor(
                    each_obj["sim_pose"], dtype=torch.float32, device=obj_xyz.device
                )
                world_xyz = (xyz_hom @ sim_pose_t.T)[:, :3].detach().cpu().numpy()
                # 与保存模型时的 target_intensity 默认值对齐（原硬编码 1.0）
                total_px += inject_into_pano(
                    depth_pano, intens_pano, world_xyz, world_to_camera_pose,
                    beam_inclinations, view.image_width, lidar_hfov, dataset.max_depth,
                    intensity_val=float(each_obj.get("inject_intensity", 0.2)),
                    sample_ratio=float(each_obj.get("inject_sample_ratio", 1.0)),
                )
            depth_numpy[0] = depth_pano
            intensity_numpy[0] = intens_pano
            logger.info(f"[inject_direct] frame={cur_frame_id}: 影响 {total_px} 个全景像素")

        spin_grid = np.stack([depth_numpy[0], intensity_numpy[0]], axis=-1).astype(
            np.float32
        )
        spin_grid = align_spin_grid_to_original_scan(spin_grid)

        point_with_intensity = pano_to_lidar_with_intensities(
            depth_numpy[0, :, :],
            intensity_numpy[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
            lidar_hfov=lidar_hfov,
        )
        gt_point_with_intensity = pano_to_lidar_with_intensities(
            gt_depth[0, :, :],
            gt_intensity[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
            lidar_hfov=lidar_hfov,
        )

        if False:  # 密度滤波
            make_raydrop = filter_pcd(point_with_intensity[:, :3])
            point_with_intensity = point_with_intensity[make_raydrop]

        if False:  # 转到baselidar系
            sensor2baselidar = gt_dynamic_model.get_sensor2baselidar(dataset.sensorid)
            points = point_with_intensity[:, :3]
            points = (
                np.pad(points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            point_with_intensity[:, :3] = points
            gt_points = gt_point_with_intensity[:, :3]
            gt_points = (
                np.pad(gt_points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            gt_point_with_intensity[:, :3] = gt_points

        # 转到 block-local 世界坐标系（原输出是传感器局部系，每帧自车都在原点，
        # 不转换的话多帧没有共同参考系）。只影响 "rendered"/"gt" 字段，在线
        # Renderer 只消费 "spin_grid"，不受影响。
        point_with_intensity[:, :3] = (
            np.pad(point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
            @ camera_to_world_pose.T
        )[:, :3]
        original_l2w = original_l2ws[idx]
        gt_point_with_intensity[:, :3] = (
            np.pad(gt_point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
            @ original_l2w.T
        )[:, :3]
        ego_pos = camera_to_world_pose[:3, 3].copy()

        # 取当前时间戳，保留三位小数，作为key
        render_timestamp = time.time()
        render_outputs[f"{render_timestamp:.3f}"] = {
            "rendered": point_with_intensity,
            "gt": gt_point_with_intensity,
            "spin_grid": spin_grid,
            "ego_pos": ego_pos,
        }

    total_render_time = time.time() - total_render_start
    fps = total_frames / total_render_time if total_render_time > 0 else 0

    logger.info(f"\n渲染完成:")
    logger.info(f"总渲染帧数: {total_frames}")
    logger.info(f"总渲染时间: {total_render_time:.4f} 秒")
    logger.info(f"平均帧率 (FPS): {fps:.4f}")

    if t_list:
        avg_frame_time = sum(t_list) / len(t_list)
        logger.info(f"单帧平均渲染时间: {avg_frame_time:.4f} 秒")

    return render_outputs


def render_sets(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    edit_obj_info,
    insert_objs,
    logger,
    save_ckpt_dir: Optional[str] = None,
    edit_yaml_src: Optional[str] = None,
    cfg_args_src: Optional[str] = None,
):
    render_cache = prepare_render_cache(
        gt_dynamic_model,
        dataset,
        iteration,
        edit_obj_info,
        logger,
    )
    if save_ckpt_dir is not None:
        # 用实际 iteration 号命名目录，让后续工具能用 searchForMaxIteration 找到
        save_iteration = iteration
        if iteration == -1:
            block_path = os.path.join(dataset.model_path, "0", str(dataset.block_id))
            if os.path.exists(block_path):
                save_iteration = searchForMaxIteration(block_path)
        save_edited_gaussian_state(
            render_cache.model_id_scene_info,
            save_ckpt_dir,
            dataset.block_id,
            save_iteration,
            edit_yaml_src=edit_yaml_src,
            cfg_args_src=cfg_args_src,
        )
        logger.info(f"[save_ckpt] 编辑后模型已保存至 {save_ckpt_dir}")
    with torch.no_grad():
        render_outputs = render_set(
            render_cache.gt_dynamic_model,
            dataset,
            "test",
            iteration,
            render_cache.valid_timestamp_model,
            render_cache.model_id_scene_info,
            render_cache.static_views,
            pipeline,
            render_cache.background,
            insert_objs,
        )
    return render_outputs


def prepare_render_cache(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    edit_obj_info,
    logger,
):
    model_id_list = [0]
    if gt_dynamic_model.get_dynamic_obj_id_list() is not None:
        model_id_list.extend(gt_dynamic_model.get_dynamic_obj_id_list())
    model_id_list = list(dict.fromkeys(model_id_list))
    edit_obj_info = remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info)
    model_id_list = delete_specified_objs(model_id_list, edit_obj_info)

    with torch.no_grad():
        static_views = None
        model_id_scene_info = {}

        for model_id in model_id_list:
            model_gaussians = GaussianModel(
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
            # lidargs 结构: {model_path}/{model_id}/{block_id}/iteration_N/
            # Scene.__init__ 里 -1 会去找 point_cloud/ 目录（3DGS 旧格式），需预先解析
            actual_iteration = iteration
            if iteration == -1:
                block_path = os.path.join(
                    dataset.model_path, str(model_id), str(dataset.block_id)
                )
                if os.path.exists(block_path):
                    actual_iteration = searchForMaxIteration(block_path)
                    logger.info(
                        f"[iteration 自动检测] model_id={model_id} "
                        f"block_id={dataset.block_id} → iteration={actual_iteration}"
                    )
                else:
                    logger.info(
                        f"[iteration 自动检测] model_id={model_id} "
                        f"block_id={dataset.block_id} 无对应 checkpoint，跳过"
                    )
                    continue
            model_scene = Scene(
                dataset,
                model_id,
                gt_dynamic_model,
                model_gaussians,
                load_iteration=actual_iteration,
                shuffle=False,
            )
            if not model_scene.init_status:
                logger.info(f"model_id {model_id} init_status is False")
                continue

            time_poses = {}
            total_views = model_scene.getTotalCameras()
            if model_id == 0:
                static_views = total_views
            for view in total_views:
                timestamp = view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = np.transpose(view.R)
                camera_pose[:3, 3] = view.T
                time_poses[timestamp] = TimePose(
                    camera_pose=camera_pose,
                    view=view,
                    valid_mask=view.img_mask,
                    gt_mask=view.original_image,
                )
            model_gaussians = move_specified_objs(
                model_id, model_gaussians, edit_obj_info
            )
            model_id_scene_info[model_id] = GaussianView(
                gaussians=model_gaussians, scene=model_scene, time_poses=time_poses
            )
            model_gaussians.eval()
        test_views = []
        test_timestamp = []
        for idx, scene_view in enumerate(static_views):
            render_timestamp = scene_view.image_name
            test_timestamp.append(int(render_timestamp))
            test_views.append(scene_view)
        valid_timestamp_model = {}
        for view in static_views:
            timestamp = view.image_name
            valid_timestamp_model[timestamp] = []
            for model_id in model_id_scene_info.keys():
                if timestamp in model_id_scene_info[model_id].time_poses:
                    valid_timestamp_model[timestamp].append(
                        model_id
                    )  # 记录每一帧timestep（0-50）下出现的obj的id

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

    return RenderCache(
        gt_dynamic_model=gt_dynamic_model,
        valid_timestamp_model=valid_timestamp_model,
        model_id_scene_info=model_id_scene_info,
        static_views=test_views,
        view_by_timestamp={view.image_name: view for view in test_views},
        background=background,
    )


def camera_pose_from_view(view: Camera) -> np.ndarray:
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.transpose(view.R)
    camera_pose[:3, 3] = view.T
    return camera_pose


def set_view_relative_pose(view: Camera, relative_pose: np.ndarray):
    """Apply a full sensor-frame delta to a cached LiDAR view."""
    relative_pose = np.asarray(relative_pose, dtype=np.float64).reshape(4, 4)
    if not hasattr(view, "_base_camera_to_world"):
        base_world_to_camera = camera_pose_from_view(view)
        view._base_camera_to_world = np.linalg.inv(base_world_to_camera)
        view._base_lidar_center = view.lidar_center.detach().clone()

    camera_to_world = view._base_camera_to_world @ relative_pose
    world_to_camera = np.linalg.inv(camera_to_world)
    view.R = world_to_camera[:3, :3].T.astype(np.float32)
    view.T = world_to_camera[:3, 3].astype(np.float32)
    view.world_view_transform = (
        torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale))
        .transpose(0, 1)
        .to(view.data_device)
    )
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(
            view.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    view.lidar_center = view.camera_center.reshape(1, 3)


def render_cached_frame(
    temporal_cache: RenderCache,
    static_cache: RenderCache,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    temporal_frame_idx: int,
    static_anchor_frame_idx: int,
    temporal_relative_pose: np.ndarray,
    static_relative_pose: np.ndarray,
    insert_objs,
):
    temporal_timestamp = str(temporal_frame_idx)
    static_timestamp = str(static_anchor_frame_idx)
    if temporal_timestamp not in temporal_cache.view_by_timestamp:
        raise KeyError(
            f"temporal frame {temporal_frame_idx} not found in temporal cache"
        )
    if static_timestamp not in static_cache.view_by_timestamp:
        raise KeyError(
            f"static anchor frame {static_anchor_frame_idx} not found in static cache"
        )
    if 0 not in static_cache.model_id_scene_info:
        raise KeyError("static model 0 not found in spatial block cache")

    # The output LiDAR view and static Gaussian model come from the block whose
    # training trajectory is spatially closest to the closed-loop request pose.
    view = static_cache.view_by_timestamp[static_timestamp]
    set_view_relative_pose(view, static_relative_pose)
    static_scene_info = static_cache.model_id_scene_info[0]
    static_time_pose = static_scene_info.time_poses[static_timestamp]
    combined_model_info = {
        0: GaussianView(
            gaussians=static_scene_info.gaussians,
            scene=static_scene_info.scene,
            time_poses={
                static_timestamp: TimePose(
                    camera_pose=camera_pose_from_view(view),
                    view=view,
                    valid_mask=static_time_pose.valid_mask,
                    gt_mask=static_time_pose.gt_mask,
                )
            },
        )
    }
    combined_model_ids = [0]

    # Dynamic objects remain tied to the original-scene timestamp. Preserve the
    # existing temporal relative-pose behavior, but expose their time pose under
    # the static anchor key so render_set can composite both cache sources.
    for model_id in temporal_cache.valid_timestamp_model[temporal_timestamp]:
        if model_id == 0:
            continue
        scene_info = temporal_cache.model_id_scene_info[model_id]
        time_pose = scene_info.time_poses[temporal_timestamp]
        object_view = time_pose.view
        set_view_relative_pose(object_view, temporal_relative_pose)
        combined_model_info[model_id] = GaussianView(
            gaussians=scene_info.gaussians,
            scene=scene_info.scene,
            time_poses={
                static_timestamp: TimePose(
                    camera_pose=camera_pose_from_view(object_view),
                    view=object_view,
                    valid_mask=time_pose.valid_mask,
                    gt_mask=time_pose.gt_mask,
                )
            },
        )
        combined_model_ids.append(model_id)

    combined_valid_timestamp_model = {
        static_timestamp: combined_model_ids,
    }

    return render_set(
        static_cache.gt_dynamic_model,
        dataset,
        "test",
        iteration,
        combined_valid_timestamp_model,
        combined_model_info,
        [view],
        pipeline,
        static_cache.background,
        insert_objs,
    )


def delete_specified_objs(model_id_list, edit_obj_info):
    if edit_obj_info is not None:
        for delete_id in edit_obj_info.delete_obj_ids:
            if delete_id in model_id_list:
                model_id_list.remove(delete_id)
    logger.info(f"model_id_list {model_id_list}")
    return model_id_list


def move_specified_objs(model_id, model_gaussians, edit_obj_info):
    transform_R = torch.eye(
        3,
        device=model_gaussians._anchor.device,
        dtype=model_gaussians._anchor.dtype,
    )
    if edit_obj_info is not None:
        for move_pair in edit_obj_info.obj_id_offset_pairs:
            if move_pair[0] == model_id:
                transform_T = torch.tensor(
                    move_pair[1],
                    device=model_gaussians._anchor.device,
                    dtype=model_gaussians._anchor.dtype,
                )
                new_anchor = (
                    model_gaussians._anchor @ transform_R.T
                ) + transform_T.reshape(1, 3)
                model_gaussians._anchor.data.copy_(new_anchor)

    return model_gaussians


def remap_instance_id_to_model_id(gt_dynamic_model, instance_id):
    if hasattr(gt_dynamic_model, "instance_id_to_model_id"):
        return gt_dynamic_model.instance_id_to_model_id(instance_id)
    return int(instance_id)


def remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info):
    if edit_obj_info is None:
        return edit_obj_info
    return EditObjInfo(
        delete_obj_ids=[remap_instance_id_to_model_id(gt_dynamic_model, x) for x in edit_obj_info.delete_obj_ids],
        obj_id_offset_pairs=[
            [remap_instance_id_to_model_id(gt_dynamic_model, pair[0]), pair[1]]
            for pair in edit_obj_info.obj_id_offset_pairs
        ],
        add_id_path_pairs=edit_obj_info.add_id_path_pairs,
        novel_poses=edit_obj_info.novel_poses,
    )








def parse_and_apply_edit_yaml(args, yaml_path: str):
    """
    Parse edit_config.yaml and map its fields to args.delete / args.move / args.add
    / args.add_waypoints / args.background_add. Also parses save_ckpt /
    output_ckpt_dir (in sync with image-side yaml format). Ignores novel_poses
    in the yaml. Ensures args.novel_poses exists (empty if absent).
    """
    if yaml_path is None:
        if not hasattr(args, "novel_poses"):
            args.novel_poses = []
        return

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"edit yaml not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    nodes = cfg.get("Nodes", {}) or {}
    rigid = nodes.get("RigidNodes", {}) or {}

    # remove -> args.remove: {"obj_ids": [...]}
    remove_section = rigid.get("remove", {}) or {}
    remove_ids = remove_section.get("instance_id", []) or []
    setattr(args, "remove", {"obj_ids": remove_ids})

    # trajectory_edit 兼容旧 trajectory 键，也支持 SMPLNodes.trajectory
    # （雷达侧无 SMPL 骨骼，统一当 RigidNodes trajectory 处理）
    # -> args.move: {"obj_id_offset_pairs": [[id, [x,y,z]], ...]}
    smpl = nodes.get("SMPLNodes", {}) or {}
    smpl_traj = smpl.get("trajectory") or {}
    traj = rigid.get("trajectory_edit") or rigid.get("trajectory") or smpl_traj or {}
    if smpl_traj and not (rigid.get("trajectory_edit") or rigid.get("trajectory")):
        logger.info("[parse_edit_yaml] SMPLNodes.trajectory → 映射为 RigidNodes trajectory_edit（雷达侧统一处理）")
    ids = traj.get("instance_id", []) or []
    move_pairs = []

    if "offset_windows" in traj:
        raw_ow = traj.get("offset_windows", []) or []
        # 判断是否 per-instance 嵌套（每元素本身是 list-of-windows）
        if raw_ow and isinstance(raw_ow[0][0], list):
            offset_windows_list = raw_ow
        else:
            offset_windows_list = [raw_ow]
        # render.py 无逐帧动画能力，用足够大的 N 计算末帧（即所有窗口完成后的最终偏移），
        # 静态写进 _anchor
        for i, iid in enumerate(ids):
            windows = offset_windows_list[i] if i < len(offset_windows_list) else []
            final_off = compute_final_offset_from_windows(windows, N=1000)
            move_pairs.append([iid, final_off])
            logger.info(f"[trajectory_edit] instance_id={iid} → 静态偏移={final_off}")
    else:
        offsets = traj.get("offset", []) or []
        if offsets and isinstance(offsets[0], (int, float)):
            offsets = [offsets]
        if offsets and len(offsets) == 1 and len(ids) > 1:
            offsets = offsets * len(ids)
        for i, iid in enumerate(ids):
            off = offsets[i] if i < len(offsets) else [0.0, 0.0, 0.0]
            move_pairs.append([iid, off])

    setattr(args, "move", {"obj_id_offset_pairs": move_pairs})

    # add -> args.add: {"obj_id_path_pairs": [(obj_class, sim_pose(4x4 list)|None, path), ...]}
    # 带 waypoints 的不进 args.add，进 args.add_waypoints，由 __main__ 用
    # create_rigid_add_per_frame_objs 生成逐帧 insert_obj（inject_direct=True）
    add_section = rigid.get("add", {}) or {}
    ref_ids = add_section.get("ref_id", []) or []
    add_objs = add_section.get("add_obj", []) or []
    add_offsets = add_section.get("offset", []) or []
    waypoints_list = add_section.get("waypoints", []) or []
    methods_list = add_section.get("method", []) or []
    time_windows_list = add_section.get("time_window", []) or []
    if isinstance(methods_list, str):
        methods_list = [methods_list]

    maxlen = max(len(add_objs), len(ref_ids), 0)
    add_pairs = []
    add_waypoints = []
    for i in range(maxlen):
        ref = ref_ids[i] if i < len(ref_ids) else None
        path = add_objs[i] if i < len(add_objs) else None
        wps = waypoints_list[i] if i < len(waypoints_list) else None
        off = add_offsets[i] if i < len(add_offsets) else None

        if wps is not None and len(wps) > 0 and path is not None:
            add_waypoints.append(
                {
                    "ref_id": ref,
                    "path": path,
                    "waypoints": wps,
                    "method": methods_list[i] if i < len(methods_list) else "linear",
                    "time_window": time_windows_list[i] if i < len(time_windows_list) else None,
                }
            )
            continue

        sim_pose = None
        if isinstance(off, list) and len(off) == 3:
            sim_pose = np.eye(4).tolist()
            sim_pose[0][3] = float(off[0])
            sim_pose[1][3] = float(off[1])
            sim_pose[2][3] = float(off[2])
        add_pairs.append((ref, sim_pose, path))
    setattr(args, "add", {"obj_id_path_pairs": add_pairs})
    setattr(args, "add_waypoints", add_waypoints)

    # Background.add：按里程/时间窗口沿自车轨迹撒静态障碍物。这里只解析字段，
    # 不算位置——依赖全局 ego 轨迹，推迟到 __main__ 统一计算。
    bg = nodes.get("Background", {}) or {}
    add_bg = bg.get("add", {}) or {}
    add_objs_bg = add_bg.get("add_obj", []) or []
    background_add = []
    if add_objs_bg:
        offsets_bg = add_bg.get("offset", []) or []
        time_windows_bg = add_bg.get("time_window", []) or []
        scale_factors_bg = add_bg.get("lidar_scale_factor", []) or []
        if isinstance(scale_factors_bg, (int, float)):
            scale_factors_bg = [float(scale_factors_bg)]
        distance_intervals_bg = add_bg.get(
            "distance_interval_m",
            add_bg.get("distance_interval", add_bg.get("spacing_m", [])),
        )
        if isinstance(distance_intervals_bg, (int, float)):
            distance_intervals_bg = [float(distance_intervals_bg)]
        distance_intervals_bg = distance_intervals_bg or []
        inject_direct_flags_bg = add_bg.get("inject_direct_points", []) or []
        if isinstance(inject_direct_flags_bg, bool):
            inject_direct_flags_bg = [inject_direct_flags_bg]
        inject_sample_ratios_bg = add_bg.get("inject_sample_ratio", []) or []
        if isinstance(inject_sample_ratios_bg, (int, float)):
            inject_sample_ratios_bg = [float(inject_sample_ratios_bg)]
        rotations_bg = add_bg.get("rotation", []) or []

        for i, ply_path in enumerate(add_objs_bg):
            background_add.append(
                {
                    "path": ply_path,
                    "offset": offsets_bg[i] if i < len(offsets_bg) else [0.0, 0.0, 0.0],
                    "time_window": time_windows_bg[i] if i < len(time_windows_bg) else [0, -1, 10],
                    "scale_factor": float(scale_factors_bg[i]) if i < len(scale_factors_bg) else 1.0,
                    "distance_interval_m": (
                        float(distance_intervals_bg[i])
                        if i < len(distance_intervals_bg) and distance_intervals_bg[i] is not None
                        else None
                    ),
                    "inject_direct": bool(inject_direct_flags_bg[i]) if i < len(inject_direct_flags_bg) else True,
                    "inject_sample_ratio": float(inject_sample_ratios_bg[i]) if i < len(inject_sample_ratios_bg) else 1.0,
                    "rotation_deg": rotations_bg[i] if i < len(rotations_bg) else [0.0, 0.0, 0.0],
                }
            )
    setattr(args, "background_add", background_add)

    # 图像侧 yaml 只写 output_ckpt_dir（指向图像模型目录）；约定只要出现它，
    # 雷达侧默认开 save_ckpt，模型存到其同级新建的 lidarckpt/，不共用目录。
    _raw_output_ckpt_dir = cfg.get("output_ckpt_dir", None) or None
    if _raw_output_ckpt_dir is not None:
        args.save_ckpt = bool(cfg.get("save_ckpt", True))
        args.output_ckpt_dir = os.path.join(
            os.path.dirname(os.path.normpath(_raw_output_ckpt_dir)), "lidarckpt"
        )
    else:
        args.save_ckpt = bool(cfg.get("save_ckpt", False))
        args.output_ckpt_dir = None


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_frames", nargs="+", type=int, default=[])
    parser.add_argument("--edit_yaml", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="chery")
    parser.add_argument("--block_size", type=int, default=50)
    args = parser.parse_args(sys.argv[1:])
    out_dir = getattr(args, "output_path", None) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, args.model_path)
    # 解析 YAML 编辑配置并映射到 args（如果提供 --edit_yaml）
    parse_and_apply_edit_yaml(args, getattr(args, "edit_yaml", None))
    save_ckpt = getattr(args, "save_ckpt", False)
    output_ckpt_dir = getattr(args, "output_ckpt_dir", None)
    if save_ckpt and output_ckpt_dir is None:
        output_ckpt_dir = os.path.join(out_dir, "edited_model")
    if save_ckpt:
        logger.info(f"[save_ckpt] 编辑后模型将保存至 {output_ckpt_dir}")
    # 注意：不传 --edit_yaml 时 parse_and_apply_edit_yaml 会把 args.novel_poses
    # 显式设成 []（见该函数），所以这里不能用 hasattr 判断"是否有 novel_poses"——
    # hasattr 恒为 True，会导致不传 edit_yaml 时 test_frames 永远不会自动补全成
    # 全部帧，渲染 0 帧却不报错。改成判断内容是否非空。
    if len(args.test_frames) == 0 and not getattr(args, "novel_poses", None):
        bin_files = [f for f in os.listdir(os.path.join(args.source_path, "lidar", "bin")) if f.endswith(".bin")]
        bin_files = sorted(bin_files, key=lambda x: int(x.split(".")[0]))
        frame_num = len(bin_files)
        args.test_frames = [x for x in range(0, frame_num, 1)]
    logger.info("Rendering " + args.model_path)

    safe_state(args.quiet)
    model_args = model.extract(args)

    if args.dataset == "chery":
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
    elif args.dataset == "zdrive":
        from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
    elif args.dataset == "chery_lidar_360":
        from scene.adaptive_dataloader import AdaptiveDataloader as GT_Dataloader
    else:
        logger.info("\nUnsupported data format.")
        sys.exit(1)

    train_frame_times = []
    if args.test_frames is not None:
        train_frame_times = args.test_frames
    else:
        for item in args.novel_poses:
            train_frame_times.append(item["frame_id"])
    block_info = {}
    for frame_id in train_frame_times:
        block_id = frame_id // args.block_size
        if block_id not in block_info:
            block_info[block_id] = []
        block_info[block_id].append(frame_id)

    edit_obj_info = EditObjInfo(
        delete_obj_ids=args.delete["obj_ids"] if hasattr(args, "delete") else [],
        obj_id_offset_pairs=(
            args.move["obj_id_offset_pairs"] if hasattr(args, "move") else []
        ),
        add_id_path_pairs=args.add["obj_id_path_pairs"] if hasattr(args, "add") else [],
        novel_poses=args.novel_poses if hasattr(args, "novel_poses") else [],
    )
    static_objs = batch_read_plys_to_insert_objs(edit_obj_info)

    # waypoints 产生带 frame_id 的逐帧 insert_obj；Background.add 产生不带
    # frame_id 的静态障碍物（可见性交给 max_depth 距离剔除）。两者 sim_pose 都在
    # 全局坐标系下算一次，每个 block 再在下面的循环里转到 block-local 坐标。
    all_train_frame_times = sorted(set(int(f) for f in train_frame_times))
    dynamic_objs_global = []
    add_waypoints = getattr(args, "add_waypoints", [])
    background_add = getattr(args, "background_add", [])
    if add_waypoints or background_add:
        scene_pose_map = load_relative_lidar_poses(args.source_path, all_train_frame_times)
        scene_l2ws = [scene_pose_map[fid] for fid in all_train_frame_times]
        cam_trajectory = compute_cam_trajectory_from_source(scene_l2ws, args.source_path)
        if cam_trajectory is not None:
            ego_trajectory_global = cam_trajectory
            logger.info(
                f"[edit_yaml] 使用前置相机轨迹摆放 add 对象"
                f"（{len(all_train_frame_times)} 帧），与相机编辑侧对齐"
            )
        else:
            ego_trajectory_global = compute_ego_trajectory(scene_l2ws)
            logger.info(
                f"[edit_yaml] 使用 LiDAR 轨迹摆放 add 对象"
                f"（{len(all_train_frame_times)} 帧），相机外参不可用"
            )
        for entry in add_waypoints:
            dynamic_objs_global.extend(
                create_rigid_add_per_frame_objs(
                    ply_path=entry["path"],
                    waypoints=entry["waypoints"],
                    method=entry["method"],
                    time_window=entry["time_window"],
                    ego_trajectory=ego_trajectory_global,
                    edit_frame_ids=all_train_frame_times,
                )
            )

        N = len(ego_trajectory_global)
        for entry in background_add:
            start_idx, end_idx, step = normalize_scene_range(entry["time_window"], N)
            spacing_m = entry["distance_interval_m"]
            if spacing_m is not None:
                selected_indices = sample_scene_indices_by_distance(
                    ego_trajectory_global, start_idx, end_idx, spacing_m
                )
            else:
                selected_indices = list(range(start_idx, end_idx, step))

            rot_matrix = Rotation.from_euler(
                "XYZ", entry["rotation_deg"], degrees=True
            ).as_matrix().astype(np.float32)
            offset = entry["offset"]
            offset_z = float(offset[2] if len(offset) > 2 else 0)

            placed = 0
            for scene_idx in selected_indices:
                ego_pos = ego_trajectory_global[scene_idx]
                sim_pose = np.eye(4, dtype=np.float32)
                sim_pose[:3, :3] = rot_matrix
                sim_pose[0, 3] = float(ego_pos[0]) + float(offset[0])
                sim_pose[1, 3] = float(ego_pos[1]) + float(offset[1])
                sim_pose[2, 3] = float(ego_pos[2]) + offset_z
                obj = read_single_ply_to_obj(
                    entry["path"], obj_class="barrier", sim_pose=sim_pose,
                    scale_factor=entry["scale_factor"],
                )
                if obj is None:
                    continue
                # 不设 frame_id：静态障碍物在世界坐标系下位置固定，每帧是否出现在
                # 该帧点云里，交给 inject_into_pano 按 max_depth 距离自然剔除。
                obj["inject_direct"] = entry["inject_direct"]
                obj["inject_sample_ratio"] = entry["inject_sample_ratio"]
                dynamic_objs_global.append(obj)
                placed += 1
            logger.info(
                f"[Background.add] {entry['path']} → 沿轨迹放置 {placed} 个静态障碍物副本"
                f"（frame {int(all_train_frame_times[start_idx]) if start_idx < len(all_train_frame_times) else start_idx}"
                f"–{int(all_train_frame_times[end_idx - 1]) if 0 <= end_idx - 1 < len(all_train_frame_times) else end_idx - 1}）"
            )

    output_paths = []
    lidar_id = 0  # 默认使用 lidar_id 0
    pose_first_block = None  # 全部编辑帧中第一个 block 的绝对 pose，作为全局坐标系原点
    ego_pos_by_frame = {}  # frame_id -> [x,y,z]，与 renders_edit/ 下 pcd 同坐标系，供视频脚本消费
    for block_id, train_frame_times in block_info.items():
        model_args.block_id = block_id  # update block id
        gt_dynamic_model = GT_Dataloader(
            model_args, train=False, train_frame_times=train_frame_times
        )
        if hasattr(args, "novel_poses"):
            gt_dynamic_model.set_novel_poses_setting(args.novel_poses)

        # 将全局坐标系下的动态 insert_obj 变换到当前 block 的局部渲染坐标系。
        #
        # 120°（Chery_Dataloader.lidar_to_world_start 恒为全场景第一帧绝对
        # 位姿，与 block 无关）和 360°（AdaptiveDataloader 用当前 block 自己的
        # 起始帧位姿，逐 block 不同）语义不同：120° 下这段变换恒为单位矩阵/no-op，
        # 360° 下才真正生效。数学上两者用同一套代码都能正确处理，不需要分支
        # （2026-08-06 确认）。
        # 已知限制：360° 多 block 场景下 ego_positions.json/点云按各 block 局部系
        # 存储，跨 block 边界 BEV 视频可能跳变——renderedit.py 原架构固有限制，
        # 非本次引入；120° 不受影响。
        objs = list(static_objs)
        if dynamic_objs_global:
            pose_current_block = gt_dynamic_model.get_lidar_to_world_start()
            if pose_first_block is None:
                pose_first_block = pose_current_block
            block_coord_transform = np.linalg.inv(pose_first_block) @ pose_current_block
            to_block_local = np.linalg.inv(block_coord_transform)
            for obj in dynamic_objs_global:
                obj_copy = dict(obj)
                sp = obj_copy.get("sim_pose")
                if sp is not None:
                    sp_np = sp.cpu().numpy().reshape(4, 4) if isinstance(sp, torch.Tensor) else np.array(sp).reshape(4, 4)
                    sp_np = to_block_local.astype(np.float32) @ sp_np
                    obj_copy["sim_pose"] = torch.tensor(sp_np, dtype=torch.float32, device="cuda")
                objs.append(obj_copy)

        render_outputs = render_sets(
            gt_dynamic_model,
            model_args,
            args.iteration,
            pipeline.extract(args),
            edit_obj_info,
            objs,
            logger,
            save_ckpt_dir=output_ckpt_dir if save_ckpt else None,
            edit_yaml_src=getattr(args, "edit_yaml", None),
            cfg_args_src=os.path.join(args.model_path, "cfg_args"),
        )
        # render_outputs 的 key 是 render_set 里用系统当前时间生成的字符串，
        # 仅用于保证字典 key 唯一，不代表真实渲染时刻；实际文件名用下面
        # sorted_frame_ids 里的真实 frame_id，跟 render_outputs 的插入顺序
        # 一一对应。点云统一收进 renders_edit/。
        sorted_frame_ids = sorted(train_frame_times)
        pcd_out_dir = os.path.join(out_dir, "renders_edit")
        os.makedirs(pcd_out_dir, exist_ok=True)
        for (key, pcd), frame_id in zip(render_outputs.items(), sorted_frame_ids):
                print(f"[Lidar Renderer] Saving rendered point cloud for frame {frame_id}...")
                render_pcd = pcd["rendered"]
                render_output_filename = os.path.join(
                    pcd_out_dir, f"ldr{lidar_id}_{frame_id:06d}.pcd"
                )
                save_pcd_binary(
                    render_output_filename,
                    render_pcd,
                )
                output_paths.append(render_output_filename)
                if "ego_pos" in pcd:
                    ego_pos_by_frame[frame_id] = [float(v) for v in pcd["ego_pos"]]
        print(f"[Lidar Renderer] Rendering completed. Outputs saved to {out_dir}")
    if ego_pos_by_frame:
        ego_pos_path = os.path.join(out_dir, "renders_edit", "ego_positions.json")
        with open(ego_pos_path, "w") as f:
            json.dump(ego_pos_by_frame, f)
        logger.info(f"[Lidar Renderer] 自车位置（与 renders_edit/ 点云同坐标系）已保存至 {ego_pos_path}")
    logger.info(f"All rendered point clouds saved to {out_dir}")

    # RigidNodes.add.waypoints 保存成真动态物体（见 bake_rigid_add_as_dynamic_object
    # 顶部说明）：只有 save_ckpt=true 且确有 waypoints 插入物体时才做，跟上面
    # "渲染时叠加"的 objs/dynamic_objs_global 逻辑完全独立、互不影响。
    if save_ckpt and add_waypoints:
        lidar_to_world_start = np.loadtxt(
            os.path.join(args.source_path, "lidar_pose", f"{all_train_frame_times[0]:06d}.txt")
        ).reshape(4, 4)

        # 标注基础取只读的 model_path（而非本次写入的 output_ckpt_dir），
        # 保证同一条命令重跑幂等，不会重复叠加物体
        _model_path_self_contained = os.path.join(args.model_path, "instances", "instances_info.json")
        source_instances_json = (
            _model_path_self_contained if os.path.exists(_model_path_self_contained)
            else os.path.join(args.source_path, "instances", "instances_info.json")
        )

        existing_ids = set()
        if os.path.exists(source_instances_json):
            with open(source_instances_json) as f:
                existing_ids = set(int(k) for k in _hugsim_expand(json.load(f)).keys())
        next_instance_id = (max(existing_ids) + 1) if existing_ids else 0

        bake_iteration = args.iteration
        if bake_iteration == -1:
            probe_block = next(iter(block_info.keys()))
            probe_path = os.path.join(output_ckpt_dir, "0", str(probe_block))
            if os.path.exists(probe_path):
                bake_iteration = searchForMaxIteration(probe_path)

        for entry in add_waypoints:
            bake_rigid_add_as_dynamic_object(
                ply_path=entry["path"],
                waypoints=entry["waypoints"],
                method=entry["method"],
                time_window=entry["time_window"],
                ego_trajectory_local=ego_trajectory_global,
                edit_frame_ids=all_train_frame_times,
                lidar_to_world_start=lidar_to_world_start,
                dataset=model_args,
                gt_data_root=output_ckpt_dir,
                source_instances_json=source_instances_json,
                block_info=block_info,
                iteration=bake_iteration,
                instance_id=next_instance_id,
            )
            next_instance_id += 1

    sys.exit(0)
