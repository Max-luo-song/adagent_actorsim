import os
import argparse
from time import time
from lidargs.arguments import ModelParams, PipelineParams
from sim_render.lidar.render import (
    prepare_render_cache,
    render_cached_frame,
)
from sim_render.lidar.render_utils import (
    get_logger,
    save_spin_grid_pcd_binary,
    CHERY_ELEVATION_DEG,
)
from sim_render.infer_utils import load_transform_matrix
import numpy as np
import io
import json
try:
    from infer_utils import pose_to_transform_matrix
except ImportError:
    from sim_render.infer_utils import pose_to_transform_matrix

LIDAR_H = 100
LIDAR_W = 1200
LIDAR_AZIMUTH_RESOLUTION_DEG = 0.1
LIDAR_AZIMUTH_START_DEG = 150.0
LIDAR_HORIZONTAL_FOV_DEG = 120.0
LIDAR_CENTERED_HORIZONTAL_FOV = False
LIDAR_CLOCKWISE_SCAN_ORDER = True
LIDAR_MAX_DEPTH = 200.0
LIDAR_METERS_PER_TICK = 0.004
FILL_EVEN_ODD_AZIMUTH_CANDIDATES = True
APPLY_INVERSE_LIDAR_EXTRINSICS = True
BEAM_INCLINATIONS = np.deg2rad(np.array(CHERY_ELEVATION_DEG, dtype=np.float32))


class LidarSpinConfig:
    def __init__(
        self,
        elevations_deg=None,
        even_azimuth_offsets_deg=None,
        odd_azimuth_offsets_deg=None,
        extrinsics=None,
        num_scans=LIDAR_W,
        azimuth_resolution_deg=LIDAR_AZIMUTH_RESOLUTION_DEG,
        azimuth_start_deg=LIDAR_AZIMUTH_START_DEG,
        horizontal_fov_deg=LIDAR_HORIZONTAL_FOV_DEG,
        centered_horizontal_fov=LIDAR_CENTERED_HORIZONTAL_FOV,
        clockwise_scan_order=LIDAR_CLOCKWISE_SCAN_ORDER,
        max_depth=LIDAR_MAX_DEPTH,
    ):
        elevations_deg = CHERY_ELEVATION_DEG if elevations_deg is None else elevations_deg
        self.elevations_rad = np.deg2rad(np.asarray(elevations_deg, dtype=np.float32))
        self.num_beams = int(len(self.elevations_rad))
        self.num_scans = int(num_scans)
        self.azimuth_resolution_deg = float(azimuth_resolution_deg)
        self.azimuth_start_deg = float(azimuth_start_deg)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.centered_horizontal_fov = bool(centered_horizontal_fov)
        self.clockwise_scan_order = bool(clockwise_scan_order)
        self.max_depth = float(max_depth)
        self.even_azimuth_offsets_deg = self._normalize_offsets(even_azimuth_offsets_deg)
        self.odd_azimuth_offsets_deg = self._normalize_offsets(odd_azimuth_offsets_deg)
        self.extrinsics = extrinsics
        self.lidar2ego = self._make_lidar2ego(extrinsics)

    def _normalize_offsets(self, offsets_deg):
        if offsets_deg is None:
            return np.zeros(self.num_beams, dtype=np.float32)
        offsets = np.asarray(offsets_deg, dtype=np.float32)
        if offsets.shape[0] < self.num_beams:
            offsets = np.pad(offsets, (0, self.num_beams - offsets.shape[0]))
        return offsets[: self.num_beams]

    def _make_lidar2ego(self, extrinsics):
        if not extrinsics:
            return None
        fields = ["x", "y", "z", "yaw", "pitch", "roll"]
        if any(extrinsics.get(field) is None for field in fields):
            return None
        return pose_to_transform_matrix(**{field: extrinsics[field] for field in fields}).astype(np.float32)


def load_lidar_spin_config(data_root, lidar_id="LDR_FRONT"):
    candidates = [
        os.path.join(data_root, "data_frame_car_info.json"),
        os.path.join(os.path.dirname(data_root), "data_frame_car_info.json"),
        os.path.join(data_root, "sensor_info.json"),
        os.path.join(os.path.dirname(data_root), "sensor_info.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)

            lidar_params = info.get("lidar_params", [])
            for param in lidar_params:
                installation = param.get("installation", {})
                inherent = param.get("inherent", {})
                intrinsics = inherent.get("intrinsics", {})
                if installation.get("lidar_id") != lidar_id and intrinsics.get("lidar_id") != lidar_id:
                    continue

                spinning = param.get("common", {}).get("spinning_lidar_params", {})
                azimuth_resolution = spinning.get(
                    "azimuth_resolution", LIDAR_AZIMUTH_RESOLUTION_DEG
                )
                cfg = LidarSpinConfig(
                    elevations_deg=intrinsics.get("elevations"),
                    even_azimuth_offsets_deg=intrinsics.get("even_azimuth_offsets"),
                    odd_azimuth_offsets_deg=intrinsics.get("odd_azimuth_offsets"),
                    extrinsics=installation.get("extrinsics"),
                    num_scans=LIDAR_W,
                    azimuth_resolution_deg=azimuth_resolution,
                )
                print(
                    "[LidarConfig] loaded "
                    f"{path}: beams={cfg.num_beams}, scans={cfg.num_scans}, "
                    f"az_res={cfg.azimuth_resolution_deg}, "
                    f"has_lidar2ego={cfg.lidar2ego is not None}"
                )
                return cfg

            if "lidar_inherent" in info:
                inherent = info["lidar_inherent"]
                cfg = LidarSpinConfig(
                    elevations_deg=inherent.get("elevations"),
                    even_azimuth_offsets_deg=inherent.get("even_azimuth_offsets"),
                    odd_azimuth_offsets_deg=inherent.get("odd_azimuth_offsets"),
                    extrinsics=info.get("lidar_installation", {}).get("extrinsics"),
                )
                print(
                    "[LidarConfig] loaded "
                    f"{path}: beams={cfg.num_beams}, scans={cfg.num_scans}, "
                    f"az_res={cfg.azimuth_resolution_deg}, "
                    f"has_lidar2ego={cfg.lidar2ego is not None}"
                )
                return cfg
        except Exception as e:
            print(f"[LidarConfig] failed to load {path}: {e}")

    cfg = LidarSpinConfig()
    print(
        "[LidarConfig] using fallback: "
        f"beams={cfg.num_beams}, scans={cfg.num_scans}, "
        f"az_res={cfg.azimuth_resolution_deg}"
    )
    return cfg


def ego_xyz_to_lidar_xyz(xyz, spin_config):
    if not APPLY_INVERSE_LIDAR_EXTRINSICS or spin_config.lidar2ego is None:
        return xyz

    ego2lidar = np.linalg.inv(spin_config.lidar2ego)
    xyz_h = np.concatenate(
        [xyz.astype(np.float32), np.ones((xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (xyz_h @ ego2lidar.T)[:, :3]


def azimuth_deg_to_scan_col(point_az_deg, azimuth_offset_deg, spin_config):
    scan_azimuth_deg = point_az_deg - azimuth_offset_deg
    if spin_config.centered_horizontal_fov:
        half_fov = spin_config.horizontal_fov_deg * 0.5
        scan_azimuth_deg = scan_azimuth_deg - spin_config.azimuth_start_deg
        scan_azimuth_deg = ((scan_azimuth_deg + 180.0) % 360.0) - 180.0
        if spin_config.clockwise_scan_order:
            col_float = (half_fov - scan_azimuth_deg) / spin_config.azimuth_resolution_deg
        else:
            col_float = (scan_azimuth_deg + half_fov) / spin_config.azimuth_resolution_deg
    else:
        if spin_config.clockwise_scan_order:
            azimuth_delta_deg = spin_config.azimuth_start_deg - scan_azimuth_deg
        else:
            azimuth_delta_deg = scan_azimuth_deg - spin_config.azimuth_start_deg
        azimuth_delta_deg = ((azimuth_delta_deg + 180.0) % 360.0) - 180.0
        col_float = azimuth_delta_deg / spin_config.azimuth_resolution_deg
    return int(round(col_float))


def parse_pcd_header(pcd_data):
    header_lines = []
    offset = 0
    for line in pcd_data.splitlines(keepends=True):
        offset += len(line)
        decoded = line.decode("utf-8", errors="replace").strip()
        header_lines.append(decoded)
        if decoded.startswith("DATA"):
            break
    else:
        raise ValueError("PCD DATA line not found")

    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0]] = parts[1:]
    return header, offset


def parse_xyz_intensity_pcd(pcd_data):
    header, payload_offset = parse_pcd_header(pcd_data)
    fields = header.get("FIELDS", [])
    data_type = header.get("DATA", [""])[0].lower()
    if fields[:4] != ["x", "y", "z", "intensity"]:
        raise ValueError(f"Unsupported PCD fields: {fields}")

    payload = pcd_data[payload_offset:]
    if data_type == "ascii":
        points = np.loadtxt(io.BytesIO(payload), dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points[:, :4]

    if data_type == "binary":
        sizes = [int(x) for x in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(x) for x in header.get("COUNT", [])]
        if sizes[:4] != [4, 4, 4, 4] or types[:4] != ["F", "F", "F", "F"] or counts[:4] != [1, 1, 1, 1]:
            raise ValueError("Only float32 x y z intensity binary PCD is supported")
        points = int(header.get("POINTS", [0])[0])
        data = np.frombuffer(payload, dtype=np.float32, count=points * len(fields))
        return data.reshape(points, len(fields))[:, :4]

    raise ValueError(f"Unsupported PCD DATA type: {data_type}")

def xyz_intensity_to_spin_grid(points, spin_config=None, is_ego_points=True):
    if spin_config is None:
        spin_config = LidarSpinConfig()

    source_xyz = points[:, :3]
    intensity = points[:, 3]
    if is_ego_points:
        lidar_xyz = ego_xyz_to_lidar_xyz(source_xyz, spin_config)
    else:
        lidar_xyz = source_xyz
    ranges = np.linalg.norm(lidar_xyz, axis=1)

    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges < spin_config.max_depth)
    source_xyz = source_xyz[valid]
    lidar_xyz = lidar_xyz[valid]
    intensity = intensity[valid]
    ranges = ranges[valid]

    x, y, z = lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2]
    point_azimuth_deg = np.mod(np.degrees(np.arctan2(x, y)), 360.0)
    alpha = np.arctan2(z, np.sqrt(x * x + y * y))
    beam_indices = np.abs(alpha[:, None] - spin_config.elevations_rad[None, :]).argmin(axis=1)
    rows = beam_indices

    in_bounds = (rows >= 0) & (rows < spin_config.num_beams)
    rows = rows[in_bounds]
    point_azimuth_deg = point_azimuth_deg[in_bounds]
    ranges = ranges[in_bounds]
    intensity = intensity[in_bounds]

    range_grid = np.zeros((spin_config.num_beams, spin_config.num_scans), dtype=np.float32)
    intensity_grid = np.zeros((spin_config.num_beams, spin_config.num_scans), dtype=np.float32)

    for point_index, (row, point_az_deg, range_m, inten) in enumerate(zip(
        rows, point_azimuth_deg, ranges, intensity
    )):
        offsets = [("even", spin_config.even_azimuth_offsets_deg[row])]
        if FILL_EVEN_ODD_AZIMUTH_CANDIDATES:
            odd_offset = spin_config.odd_azimuth_offsets_deg[row]
            if not np.isclose(odd_offset, offsets[0][1]):
                offsets.append(("odd", odd_offset))

        for offset_kind, azimuth_offset_deg in offsets:
            col = azimuth_deg_to_scan_col(point_az_deg, azimuth_offset_deg, spin_config)
            if col < 0 or col >= spin_config.num_scans:
                continue
            old_range = range_grid[row, col]
            if old_range == 0.0 or range_m < old_range:
                range_grid[row, col] = range_m
                intensity_grid[row, col] = inten

    spin_grid = np.stack([range_grid, intensity_grid], axis=-1).astype(np.float32)
    return spin_grid


def spin_grid_to_pcd_binary_bytes(spin_grid):
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
    return header.encode("ascii") + np.ascontiguousarray(grid).tobytes()


def pcd_xyz_intensity_to_spin_pcd_binary_bytes(points, spin_config=None):
    spin_grid = xyz_intensity_to_spin_grid(points, spin_config, False)
    # log_intensity_stats("3dgs_spin_grid_intensity", spin_grid[..., 1])
    # log_intensity_stats("3dgs_spin_grid_valid_intensity", spin_grid[..., 1][spin_grid[..., 0] > 0])
    return spin_grid_to_pcd_binary_bytes(spin_grid)

def save_replaced_sensor_data(file_path: str, spin_pcd_data):
    with open(file_path, "wb") as f:
        f.write(spin_pcd_data)

def log_intensity_stats(name, values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"[{name}] empty")
        return

    qs = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    zero_ratio = np.mean(arr == 0)
    one_ratio = np.mean(arr == 1)
    print(
        f"[{name}] n={arr.size} "
        f"min={qs[0]:.3f} p1={qs[1]:.3f} p5={qs[2]:.3f} "
        f"p25={qs[3]:.3f} p50={qs[4]:.3f} p75={qs[5]:.3f} "
        f"p95={qs[6]:.3f} p99={qs[7]:.3f} max={qs[8]:.3f} "
        f"mean={arr.mean():.3f} zero={zero_ratio:.2%} one={one_ratio:.2%} "
        f"unique~={len(np.unique(arr[:min(arr.size, 200000)]))}"
    )

class Renderer:
    """
    初始化模型、数据集，并暴露 render_single_frame() 给外部脚本实时调用
    """

    def __init__(self, lidar_checkpoint_path: str, source_path: str, output_dir="./outputs"):
        parser = argparse.ArgumentParser(description="Lidar Renderer Parameters")
        self.model_params = ModelParams(parser)
        self.pipeline_params = PipelineParams(parser)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset = 'chery'
        self.iteration = 3000
        self.lidar_checkpoint_path = lidar_checkpoint_path
        self.spatial_block_hysteresis_m = 2.0
        self.current_spatial_block_id = None
        self.quiet = True

        # set some runtime overrides and produce the final params object via extract()
        # 首先从 parser 获取带默认值的 Namespace，然后覆盖我们需要的项
        args_ns = parser.parse_args([])  # use registered defaults
        args_ns.source_path = source_path
        args_ns.model_path = lidar_checkpoint_path
        args_ns.max_depth = 100.0
        args_ns.output_path = output_dir

        # extract() 会返回一个 group (没有下划线的属性)，和主脚本行为一致
        self.model_params = self.model_params.extract(args_ns)
        self.pipeline_params = self.pipeline_params.extract(args_ns)

        # 初始化 logger（使用最终的 model_path）
        self.logger = get_logger(output_dir, args_ns.model_path)
        self.frame_num = self.count_lidar_frames()
        self.load_training_block_info()
        self.load_frame_poses()
        self.render_cache_by_block = {}
        self.preload_all_blocks()

    def count_lidar_frames(self):
        bin_dir = os.path.join(self.model_params.source_path, "lidar", "bin")
        return len([f for f in os.listdir(bin_dir) if f.endswith(".bin")])

    def get_gt_dataloader_cls(self):
        if self.dataset == "chery":
            from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
        elif self.dataset == "zdrive":
            from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
        else:
            raise ValueError(f"[Lidar Renderer] Unknown dataset: {self.dataset}")
        return GT_Dataloader

    @staticmethod
    def _normalize_block_frames(raw_mapping, field_name):
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise ValueError(f"{field_name} must be a non-empty object")
        result = {}
        for raw_block_id, raw_frames in raw_mapping.items():
            block_id = int(raw_block_id)
            frames = sorted({int(frame) for frame in raw_frames})
            if not frames:
                raise ValueError(f"{field_name}[{block_id}] is empty")
            result[block_id] = frames
        return result

    def load_training_block_info(self):
        """Load the exact block partition persisted by model training."""
        block_info_path = os.path.join(self.lidar_checkpoint_path, "block_info.json")
        if not os.path.isfile(block_info_path):
            raise FileNotFoundError(
                "LiDAR checkpoint is missing training block metadata: "
                f"{block_info_path}"
            )
        with open(block_info_path, "r", encoding="utf-8") as block_file:
            block_info = json.load(block_file)

        self.block_frames_with_extend = self._normalize_block_frames(
            block_info.get("block_time_with_extend"),
            "block_time_with_extend",
        )
        self.block_frames_without_extend = self._normalize_block_frames(
            block_info.get("block_time_without_extend"),
            "block_time_without_extend",
        )
        raw_block_rects = block_info.get("block_id_with_rect", {})
        if raw_block_rects in (None, []):
            raw_block_rects = {}
        if not isinstance(raw_block_rects, dict):
            raise ValueError("block_id_with_rect must be an object or an empty list")
        self.block_rects = {
            int(block_id): np.asarray(rect, dtype=np.float64)
            for block_id, rect in raw_block_rects.items()
        }

        missing_training_blocks = (
            set(self.block_frames_without_extend)
            - set(self.block_frames_with_extend)
        )
        if missing_training_blocks:
            raise ValueError(
                "temporal blocks have no trained cache: "
                f"{sorted(missing_training_blocks)}"
            )
        for block_id, frames in self.block_frames_without_extend.items():
            missing_frames = set(frames) - set(
                self.block_frames_with_extend[block_id]
            )
            if missing_frames:
                raise ValueError(
                    f"temporal block {block_id} contains frames absent from its "
                    f"trained cache: {sorted(missing_frames)}"
                )

        self.frame_to_temporal_block = {}
        for block_id, frames in self.block_frames_without_extend.items():
            for frame_id in frames:
                previous = self.frame_to_temporal_block.setdefault(frame_id, block_id)
                if previous != block_id:
                    raise ValueError(
                        f"frame {frame_id} belongs to temporal blocks "
                        f"{previous} and {block_id}"
                    )
        if not self.frame_to_temporal_block:
            raise ValueError("block_info.json contains no temporal frame ownership")
        self.available_temporal_frames = np.asarray(
            sorted(self.frame_to_temporal_block), dtype=np.int64
        )
        print(
            "[LidarBlockSelect] loaded training block_info "
            f"{block_info_path}: blocks={sorted(self.block_frames_with_extend)}, "
            f"temporal_frames={len(self.available_temporal_frames)}"
        )

    def load_frame_poses(self):
        required_frames = set(self.frame_to_temporal_block)
        for frames in self.block_frames_with_extend.values():
            required_frames.update(frames)
        self.frame_pose_by_id = {}
        for frame_id in sorted(required_frames):
            pose_path = os.path.join(
                self.model_params.source_path,
                "lidar_pose",
                f"{frame_id:06d}.txt",
            )
            if not os.path.isfile(pose_path):
                raise FileNotFoundError(
                    f"block_info.json references missing LiDAR pose: {pose_path}"
                )
            self.frame_pose_by_id[frame_id] = load_transform_matrix(txt_path=pose_path)

    def nearest_temporal_frame(self, frame_idx):
        frame_idx = int(frame_idx)
        position = int(np.argmin(np.abs(self.available_temporal_frames - frame_idx)))
        return int(self.available_temporal_frames[position])

    @staticmethod
    def _xy_distance_to_rect(xy, rect):
        if rect.shape != (4,):
            raise ValueError(f"invalid block rectangle: {rect}")
        dx = max(float(rect[0] - xy[0]), 0.0, float(xy[0] - rect[2]))
        dy = max(float(rect[1] - xy[1]), 0.0, float(xy[1] - rect[3]))
        return float(np.hypot(dx, dy))

    def _block_distance(self, block_id, request_xy):
        if block_id in self.block_rects:
            return self._xy_distance_to_rect(request_xy, self.block_rects[block_id])
        frame_xy = np.asarray(
            [
                self.frame_pose_by_id[idx][:2, 3]
                for idx in self.block_frames_with_extend[block_id]
            ]
        )
        return float(np.linalg.norm(frame_xy - request_xy, axis=1).min())

    def select_spatial_block(self, request_pose):
        request_xy = np.asarray(request_pose[:2, 3], dtype=np.float64)
        distances = {
            block_id: self._block_distance(block_id, request_xy)
            for block_id in self.block_frames_with_extend
        }
        best_block = min(distances, key=lambda block_id: (distances[block_id], block_id))
        current = self.current_spatial_block_id
        if current not in distances:
            selected = best_block
        elif distances[current] > distances[best_block] + self.spatial_block_hysteresis_m:
            selected = best_block
        else:
            selected = current
        self.current_spatial_block_id = selected

        candidates = self.block_frames_with_extend[selected]
        anchor_frame = min(
            candidates,
            key=lambda frame_id: np.linalg.norm(
                self.frame_pose_by_id[frame_id][:2, 3] - request_xy
            ),
        )
        anchor_distance = float(
            np.linalg.norm(self.frame_pose_by_id[anchor_frame][:2, 3] - request_xy)
        )
        return selected, anchor_frame, anchor_distance, distances

    def preload_all_blocks(self):
        GT_Dataloader = self.get_gt_dataloader_cls()
        print(
            "[Lidar Renderer] Preloading "
            f"{len(self.block_frames_with_extend)} LiDAR blocks from block_info.json..."
        )

        for block_id, train_frame_times in self.block_frames_with_extend.items():
            print(
                f"[Lidar Renderer] Preloading block {block_id}: "
                f"{len(train_frame_times)} frames, "
                f"range={train_frame_times[0]}-{train_frame_times[-1]}"
            )
            self.model_params.block_id = block_id
            gt_dynamic_model = GT_Dataloader(
                self.model_params,
                train=False,
                train_frame_times=train_frame_times,
            )
            self.render_cache_by_block[block_id] = prepare_render_cache(
                gt_dynamic_model,
                self.model_params,
                self.iteration,
                edit_obj_info=None,
                logger=self.logger,
            )

        print("[Lidar Renderer] All LiDAR blocks preloaded.")

    def find_nearest_frame(self, pose, frame_num, source_path):
        """
        找到当前pose对应的最近帧index，以及相对变换矩阵
        """
        min_dist = float('inf')
        min_idx = -1
        min_transform = None

        for idx in range(frame_num):
            lidar_pose_txt = os.path.join(source_path, "lidar_pose", f"{idx:06d}.txt")
            frame_transform = load_transform_matrix(txt_path=lidar_pose_txt)
            dist = np.linalg.norm(pose[0:3, 3] - frame_transform[0:3, 3])
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
                min_transform = frame_transform

        rel_transform = np.linalg.inv(min_transform) @ pose
        return min_idx, rel_transform
    
    def find_frame_by_timestamp(self, pose, frame_idx, source_path):
        """
        根据外部按时间戳算出的 frame_idx 找帧，并计算相对变换矩阵。
        find_nearest_frame 保留作为旧接口/备用逻辑。
        """
        lidar_pose_txt = os.path.join(source_path, "lidar_pose", f"{frame_idx:06d}.txt")
        frame_transform = load_transform_matrix(txt_path=lidar_pose_txt)
        rel_transform = np.linalg.inv(frame_transform) @ pose
        return frame_idx, rel_transform

    def render_single_frame(self, pose, lidar_id=0, frame_idx=None):
        """ Render LiDAR data from the given pose message.
        Args:
            pose: LiDAR渲染定义的世界坐标系下的novel pose
            lidar_id: LiDAR ID
            frame_idx: 如果传入，直接使用该帧；否则使用 find_nearest_frame

        Returns:

        Description:
            优先使用时间戳得到的 frame_idx（获取动态障碍物），计算新pose和该帧的相对变换
            渲染新pose下的LiDAR点云
        """
        output_paths = []
        self.lidar_id = lidar_id
        print(f"[Lidar Renderer] Rendering for LiDAR ID: {self.lidar_id}")
        if frame_idx is None:
            frame_idx, _ = self.find_nearest_frame(
                pose, self.frame_num, self.model_params.source_path
            )
        frame_idx = self.nearest_temporal_frame(frame_idx)
        temporal_block_id = self.frame_to_temporal_block[frame_idx]
        temporal_frame_pose = self.frame_pose_by_id[frame_idx]
        temporal_relative_pose = np.linalg.inv(temporal_frame_pose) @ pose

        (
            spatial_block_id,
            static_anchor_frame_idx,
            static_anchor_distance,
            block_distances,
        ) = self.select_spatial_block(pose)
        static_anchor_pose = self.frame_pose_by_id[static_anchor_frame_idx]
        static_relative_pose = np.linalg.inv(static_anchor_pose) @ pose

        level = "WARNING" if temporal_block_id != spatial_block_id else "INFO"
        print(
            f"[LidarBlockSelect][{level}] temporal_frame={frame_idx}, "
            f"temporal_block={temporal_block_id}, spatial_block={spatial_block_id}, "
            f"static_anchor={static_anchor_frame_idx}, "
            f"static_anchor_distance_m={static_anchor_distance:.3f}, "
            f"block_distances_m={block_distances}"
        )
        self.model_params.block_id = spatial_block_id
        render_outputs = render_cached_frame(
            self.render_cache_by_block[temporal_block_id],
            self.render_cache_by_block[spatial_block_id],
            self.model_params,
            self.iteration,
            self.pipeline_params,
            frame_idx,
            static_anchor_frame_idx,
            temporal_relative_pose,
            static_relative_pose,
            insert_objs=None,
        )
        for key, pcd in render_outputs.items():
            timestamp = float(key)

            print (f"[Lidar Renderer] Saving rendered spin grid for timestamp {timestamp}...")
            render_pcd = pcd["spin_grid"]
            render_output_filename = os.path.join(
                self.output_dir,
                f"ldr{lidar_id}_{frame_idx}_{timestamp}.pcd",
            )
            save_spin_grid_pcd_binary(
                render_output_filename,
                render_pcd,
            )

            output_paths.append(render_output_filename)
        print(f"[Lidar Renderer] Rendering completed. Outputs saved to {self.output_dir}")
        return output_paths

if __name__ == "__main__":
    # Example usage
    renderer = Renderer(
        lidar_checkpoint_path="/nas_thoru/users/yangtao/qcraft/processed/test/20251025_163358_QCOYSD504206_1595_1610",
        source_path="/nas_thoru/users/yangtao/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610",
        output_dir="./realtime_output"
    )

    # Example pose (4x4 transformation matrix)
    example_pose = np.array([
        [1, 0, 0, 7987.086396549613],
        [0, 1, 0, -2964.653905917249],
        [0, 0, 1, -396.08292995686327],
        [0, 0, 0, 1]
    ])

    renderer.render_single_frame(example_pose)
