
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

pass  # Path is provided by the portable boundary above
import json
import numpy as np
import argparse
from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict


@dataclass
class VehiclePose:
    """单条车辆位姿观测"""
    timestamp: float       # 时间戳 (秒), 来自 header.timestamp / 1e6
    x: float               # 世界坐标 x
    y: float               # 世界坐标 y
    z: float               # 世界坐标 z
    yaw: float             # 朝向角 (弧度)
    pitch: float
    roll: float


@dataclass
class ObjectObs:
    """单条目标观测"""
    timestamp: float       # 时间戳 (秒), 来自 object['timestamp']
    obj_id: str            # track ID, 如 "4684"
    obj_type: str          # 原始类型, 如 "OT_VEHICLE"
    # 世界坐标系下的状态
    x: float               # bounding_box.x (中心点)
    y: float               # bounding_box.y
    z: float               # camera_mea.main_source.pos.z + 0.5 * bbox3d_height
    yaw: float             # 朝向角 (弧度)
    yaw_rate: float        # 角速度 (rad/s)
    # 速度 (m/s, 世界系)
    vx: float
    vy: float
    # 3D 框尺寸
    length: float
    width: float
    height: float


@dataclass
class SmoothedState:
    """卡尔曼滤波后单帧的平滑状态"""
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    length: float
    width: float
    height: float


@dataclass
class SmoothedTrack:
    """一条经卡尔曼滤波平滑后的目标轨迹"""
    obj_id: str
    obj_type: str
    states: List[SmoothedState]   # 按时间升序排列
    t_first: float
    t_last: float


@dataclass
class ObjBbox:
    """目标对象的bounding box标注信息"""
    timestamp: float
    obj_id: str
    obj_type:str
    x: float
    y: float
    z: float
    yaw: float    
    length: float
    width: float
    height: float


TYPE_MAP = {
    "OT_PEDESTRIAN": "Pedestrian",

    "OT_MOTORCYCLIST": "Motorcycle",
    "OT_CYCLIST": "Bicycle",
    "OT_TRICYCLIST": "Bicycle",
    
    "OT_VEHICLE": "Vehicle",               
    "OT_LARGE_VEHICLE": "Vehicle",

    "OT_BARRIER": "TrafficBarrier",
    "OT_BARRIER_ANTI_COLLISION_BUCKET": "TrafficBarrier",
    "OT_BARRIER_ANTI_COLLISION_POST": "TrafficBarrier",
    "OT_BARRIER_CEMENT_PILLAR": "TrafficBarrier",
    "OT_CONE": "TrafficBarrier",           # 锥桶可视为临时交通屏障
    "OT_WARNING_TRIANGLE": "TrafficBarrier", # 三角警示牌

    # 归入 Unknown（目标列表无对应）
    "OT_FOD": "Unknown",                   # 异物碎片
    "OT_VEGETATION": "Unknown",            # 植被
    "OT_UNKNOWN_STATIC": "Unknown",
    "OT_UNKNOWN_MOVABLE": "Unknown",
}


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def world_to_vehicle(obj_x: float, obj_y: float, obj_z: float, obj_yaw: float,
                     veh_pose: VehiclePose) -> tuple:
    """
    将目标的世界坐标 + 朝向转换到车辆坐标系.

    位置:  X_veh = R^T · (X_world - t)
    朝向:  yaw_veh = yaw_world - yaw_vehicle  (归一化到 [-π, π])
    """
    t = np.array([veh_pose.x, veh_pose.y, veh_pose.z])
    R = euler_to_rotation_matrix(veh_pose.roll, veh_pose.pitch, veh_pose.yaw)
    R_inv = R.T

    pos_world = np.array([obj_x, obj_y, obj_z])
    pos_veh = R_inv @ (pos_world - t)

    yaw_veh = float((obj_yaw - veh_pose.yaw + np.pi) % (2 * np.pi) - np.pi)

    return pos_veh[0], pos_veh[1], pos_veh[2], yaw_veh


def _normalize_angle(a: float) -> float:
    """将角度归一化到 [-π, π]"""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class ConstantVelocityKF:
    """
    恒速度模型卡尔曼滤波器.

    状态: [pos_0, ..., pos_{d-1}, vel_0, ..., vel_{d-1}]  (2d 维)
    观测: [pos_0, ..., pos_{d-1}]                         (d 维)

    仅观测位置，速度由滤波器隐式估计.
    """

    def __init__(self,
                 pos_dim: int,
                 init_meas: np.ndarray,
                 init_t: float,
                 q_pos: float = 0.1,     # 位置过程噪声
                 q_vel: float = 1.0,     # 速度过程噪声
                 r_pos: float = 0.05):   # 位置观测噪声
        """
        Args:
            pos_dim:    位置维度 (如 4 表示 x,y,z,yaw)
            init_meas:  初始观测向量, shape = (pos_dim,)
            init_t:     初始时间戳
            q_pos/q_vel: 过程噪声系数 (越大 = 越信任观测, 越小 = 越信任模型)
            r_pos:      观测噪声系数
        """
        self.pos_dim = pos_dim
        self.dim = 2 * pos_dim   # 状态总维度

        # ---- 初始状态: x = [meas, 0, ..., 0] ----
        self.x = np.zeros(self.dim)
        self.x[:pos_dim] = init_meas

        # ---- 初始协方差 ----
        self.P = np.eye(self.dim) * 0.5

        # ---- 观测矩阵: H = [I_d, 0_d] ----
        self.H = np.zeros((pos_dim, self.dim))
        self.H[:, :pos_dim] = np.eye(pos_dim)

        # ---- 噪声参数 ----
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R = np.eye(pos_dim) * r_pos

        self.last_t = init_t

        # RTS 反向平滑所需中间矩阵 (在 predict 中填充)
        self.F_saved = np.eye(self.dim)
        self.x_pred  = self.x.copy()
        self.P_pred  = self.P.copy()

    def predict(self, t: float):
        """
        从 last_t 预测到 t 时刻.

        F 矩阵在 dt 上使用恒速度模型:
            pos_{t+dt} = pos_t + vel_t * dt
            vel_{t+dt} = vel_t
        """
        dt = t - self.last_t
        if dt <= 0:
            return

        # ---- 状态转移矩阵 F ----
        F = np.eye(self.dim)
        for i in range(self.pos_dim):
            F[i, self.pos_dim + i] = dt

        # ---- 过程噪声 Q (离散化白噪声) ----
        dt2 = dt * dt
        G = np.zeros((self.dim, self.pos_dim * 2))
        for i in range(self.pos_dim):
            G[i, i] = dt2 * 0.5
            G[self.pos_dim + i, i] = dt
            G[i, self.pos_dim + i] = dt2 * 0.5
            G[self.pos_dim + i, self.pos_dim + i] = dt

        Q_block = np.diag([self.q_pos] * self.pos_dim + [self.q_vel] * self.pos_dim)
        Q = G @ Q_block @ G.T

        # ---- 预测 ----
        self.F_saved = F.copy()            # 状态转移矩阵 (给 RTS 用)
        self.x = F @ self.x                # 先做预测
        self.P = F @ self.P @ F.T + Q
        self.x_pred = self.x.copy()        # 预测后状态 (update 前), 给 RTS 用
        self.P_pred = self.P.copy()        # 预测后协方差 (update 前), 给 RTS 用
        self.last_t = t

    def update(self, measurement: np.ndarray):
        """
        用观测更新状态.

        Args:
            measurement: 观测向量, shape = (pos_dim,)
        """
        # 计算创新
        z = measurement
        y = z - self.H @ self.x

        # 如果状态中有角度 (位置最后一维是 yaw 时作归一化)
        # 这里不硬编码, 让调用方自行处理; update 保持通用

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P


class ConstantAccelerationKF:
    """
    恒加速度模型卡尔曼滤波器.

    状态: [pos_0, ..., pos_{d-1}, vel_0, ..., vel_{d-1}, acc_0, ..., acc_{d-1}]  (3d 维)
    观测: [pos_0, ..., pos_{d-1}]                                                  (d 维)

    pos_{t+dt} = pos_t + vel_t·dt + ½·acc_t·dt²
    vel_{t+dt} = vel_t + acc_t·dt
    acc_{t+dt} = acc_t + noise          ← 加速度随机游走

    仅观测位置，速度和加速度由滤波器隐式估计.
    适合 cut-in、加减速等速度变化的场景.
    """

    def __init__(self,
                 pos_dim: int,
                 init_meas: np.ndarray,
                 init_t: float,
                 q_pos: float = 0.1,
                 q_vel: float = 1.0,
                 q_acc: float = 3.0,
                 r_pos: float = 0.05):
        self.pos_dim = pos_dim
        self.dim = 3 * pos_dim   # [pos..., vel..., acc...]

        # ---- 初始状态: x = [meas, 0, ..., 0] ----
        self.x = np.zeros(self.dim)
        self.x[:pos_dim] = init_meas

        # ---- 初始协方差 ----
        self.P = np.eye(self.dim) * 0.5

        # ---- 观测矩阵: H = [I_d, 0_d, 0_d] ----
        self.H = np.zeros((pos_dim, self.dim))
        self.H[:, :pos_dim] = np.eye(pos_dim)

        # ---- 噪声参数 ----
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_acc = q_acc
        self.R = np.eye(pos_dim) * r_pos

        self.last_t = init_t

        # RTS 反向平滑所需中间矩阵
        self.F_saved = np.eye(self.dim)
        self.x_pred  = self.x.copy()
        self.P_pred  = self.P.copy()

    def predict(self, t: float):
        """恒加速度模型预测."""
        dt = t - self.last_t
        if dt <= 0:
            return

        dt2 = dt * dt

        # ---- 状态转移矩阵 F ----
        # 每维: pos += vel*dt + 0.5*acc*dt², vel += acc*dt, acc 不变
        F = np.eye(self.dim)
        for i in range(self.pos_dim):
            p = i
            v = self.pos_dim + i
            a = 2 * self.pos_dim + i
            F[p, v] = dt
            F[p, a] = 0.5 * dt2
            F[v, a] = dt

        # ---- 过程噪声 Q (加速度被白噪声驱动) ----
        # G 映射: 每维的过程噪声 → 状态, G = [dt²/2, dt, 1]^T
        G = np.zeros((self.dim, self.pos_dim))
        for i in range(self.pos_dim):
            G[i, i] = 0.5 * dt2
            G[self.pos_dim + i, i] = dt
            G[2 * self.pos_dim + i, i] = 1.0

        Q = G @ (np.eye(self.pos_dim) * self.q_acc) @ G.T

        # ---- 预测 ----
        self.F_saved = F.copy()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.last_t = t

    def update(self, measurement: np.ndarray):
        """标准 KF 更新 (与 CV 相同)."""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P


class BoxExtractor:
    """
    从 objects_proto JSON 中提取 3D 目标框。

    object 的处理顺序：
        1. 按图像帧时间戳从原始 track 中选择最近观测
        2. 对采样后的 x/y/z/yaw 做高斯平滑
        3. 转换到车辆坐标系
        4. 写入 label_pred/

    使用:
        extractor = BoxExtractor(parsed_data_root, scene_id, (start, end))
        extractor.load()                     # 读取 JSON
        extractor.smooth()                   # 清洗并准备数据
        extractor.write_labels()             # 最近邻采样 + 高斯平滑 + 写入
    """

    def __init__(self,
                 parsed_data_root: str,
                 scene_id: str,
                 time_range: tuple):
        slice_start, slice_end = time_range
        scene_id_s_e = f"{scene_id}_{slice_start}_{slice_end}"

        # ---- 路径 ----
        parsed_scene_dir = Path(parsed_data_root) / scene_id
        self.obj_json_dir   = parsed_scene_dir / f"lite_msg_{slice_start}_{slice_end}" / "objects_proto"
        self.pose_json_dir   = parsed_scene_dir / f"lite_msg_{slice_start}_{slice_end}" / "pose_proto"
        self.img_output_dir = parsed_scene_dir / scene_id_s_e
        self.frame_json_path = self.img_output_dir / "data_frame_seq.json"
        self.label_dir       = self.img_output_dir / "label_pred"
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # ---- 中间数据 ----
        self.veh_poses: List[VehiclePose]           = []
        self.objects:   List[ObjectObs]              = []
        self.tracks:    Dict[str, List[ObjectObs]]   = {}   # obj_id → 观测序列

        # 车辆位姿和“按输出帧采样后”的 object 平滑结果
        self.smoothed_veh_poses: List[VehiclePose] = []
        self.smoothed_tracks: Dict[str, List[ObjBbox]] = {}

        # 图像帧序列 (load 时填充)
        self.data_frame_seq: List[dict] = []

    # ========== 步骤 1: 数据读取 ==========

    def load(self):
        """读取 objects_proto JSON, 填充 veh_poses / objects / tracks."""
        self._load_frame_seq()
        self._load_ego_poses()
        self._load_object_observations()
        self._group_by_id()
        self._print_summary()

    def _load_frame_seq(self):
        """读取 data_frame_seq.json → self.data_frame_seq"""
        with open(self.frame_json_path, 'r', encoding='utf-8') as f:
            data = _hugsim_expand(json.load(f))
        self.data_frame_seq = [dic for dic in data['data_frame_seq_items']]

    def _load_ego_poses(self):
        """
        遍历 pose_proto/*.json, 提取车辆位姿.
        位姿:  message['pose'] → {x, y, z, yaw, pitch, roll}
        时间戳: message['header']['timestamp'] (微秒字符串) → 秒
        """
        for json_file in sorted(self.pose_json_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = _hugsim_expand(json.load(f))
        
            msg = data.get('message', {})
            position = msg.get('pos_smooth')
            
            self.veh_poses.append(VehiclePose(
                timestamp = msg['timestamp'],
                x = position['x'],
                y = position['y'],
                z = position['z'],
                yaw = msg['yaw'],
                pitch = msg['pitch'],
                roll = msg['roll']
            ))
        
        self.veh_poses.sort(key=lambda p: p.timestamp)

    def _load_object_observations(self):
        """
        遍历 objects_proto/*.json, 提取目标观测.

        读取原则与 transfer_object3dbox_info 一致:
          - 必须有 camera_mea.main_source.pos (否则无法确定 z)
          - 必须有 bbox3d_height
          - x, y 取自 bounding_box
          - z = camera_mea.main_source.pos.z + 0.5 * bbox3d_height
        """
        for json_file in sorted(self.obj_json_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = _hugsim_expand(json.load(f))

            objects = data.get('message', {}).get('objects', [])

            for obj in objects:
                ts = obj.get('timestamp')
                bbox = obj.get('bounding_box', {})
                bbox3d_h = obj.get('bbox3d_height')

                object_pos = obj.get('camera_mea', {}).get('main_source', {}).get('pos', None)
                if object_pos is None or bbox3d_h is None or ts is None or not bbox:
                    continue
                
                vel = obj.get('vel', {})

                self.objects.append(ObjectObs(
                    timestamp = float(ts),
                    obj_id    = str(obj.get('id', '')),
                    obj_type  = TYPE_MAP.get(obj.get('type', ''), 'unknown'),
                    x         = bbox.get('x', 0),
                    y         = bbox.get('y', 0),
                    z         = object_pos['z'] + 0.5 * bbox3d_h,
                    yaw       = obj.get('yaw', 0),
                    yaw_rate  = obj.get('yaw_rate', 0),
                    vx        = vel.get('x', 0),
                    vy        = vel.get('y', 0),
                    length    = bbox.get('length', 0),
                    width     = bbox.get('width', 0),
                    height    = bbox3d_h,
                ))

        self.objects.sort(key=lambda o: o.timestamp)

    def _group_by_id(self):
        """将 objects 按 obj_id 分组 → self.tracks, 每组按时间排序."""
        groups: Dict[str, List[ObjectObs]] = defaultdict(list)
        for obs in self.objects:
            groups[obs.obj_id].append(obs)
        for obj_id in groups:
            groups[obj_id].sort(key=lambda o: o.timestamp)
        self.tracks = dict(groups)

    def _print_summary(self):
        n_tracks = len(self.tracks)
        avg_len = len(self.objects) / n_tracks if n_tracks > 0 else 0
        print(f"[INFO] 车辆位姿: {len(self.veh_poses)}  条")
        print(f"[INFO] 目标观测: {len(self.objects)}  条")
        print(f"[INFO] 目标 track: {n_tracks} 条, 平均每 track {avg_len:.1f} 帧")

    def clean_height_zeros(self, max_zero_ratio: float = 0.3):
        """
        清洗各 object track 中的 height=0 观测:

        1. 若某 track 的 height=0 占比 > max_zero_ratio → 整条 track 删除
        2. 首尾连续的 0 值 → 裁剪掉
        3. 中间孤立的 0 值 → 用整条 track 的非零中位数替换

        Args:
            max_zero_ratio: 零值占比阈值, 超过则删 track, 默认 0.3 (30%)
        """
        drop_ids = []
        trim_count = 0
        fill_count = 0

        for obj_id, obs_list in self.tracks.items():
            heights = np.array([o.height for o in obs_list])
            n_total = len(heights)
            n_zero = int((heights == 0.0).sum())

            # ---- 1. 零值占比过高 → 整条删除 ----
            if n_zero / n_total > max_zero_ratio:
                drop_ids.append(obj_id)
                continue

            # ---- 2. 裁剪首尾连续零值 ----
            first_nonzero = 0
            while first_nonzero < n_total and heights[first_nonzero] == 0.0:
                first_nonzero += 1
            last_nonzero = n_total - 1
            while last_nonzero >= 0 and heights[last_nonzero] == 0.0:
                last_nonzero -= 1

            if first_nonzero > 0 or last_nonzero < n_total - 1:
                trim_count += (first_nonzero + (n_total - 1 - last_nonzero))
                obs_list[:] = obs_list[first_nonzero:last_nonzero + 1]
                heights = heights[first_nonzero:last_nonzero + 1]

            # ---- 3. 中间零值用非零中位数填充 ----
            if len(heights[heights > 0]) > 0:
                median_val = np.median(heights[heights > 0])
            else:
                median_val = 1.0  # 全部都是 0 但占比未超阈值, 用默认值

            for i, o in enumerate(obs_list):
                if o.height == 0.0:
                    o.height = float(median_val)
                    fill_count += 1

        # 删除整条 track
        for oid in drop_ids:
            del self.tracks[oid]

        print(f"[INFO] height 零值清洗: 删除 {len(drop_ids)} 条 track, "
              f"裁剪首尾 {trim_count} 帧, 中位数填充中间 {fill_count} 帧")

    MIN_TRACK_LEN = 3

    def smooth(self):
        """
        对 vehicle pose 时序和每条 object track 分别做卡尔曼滤波,
        填充 self.smoothed_veh_poses 和 self.smoothed_tracks.
        """
        print("[INFO] 清洗 height 零值...")
        self.clean_height_zeros()
        # print("[INFO] 开始卡尔曼滤波平滑...")
        # 直接使用pose proto, 里面的ego vehcile位姿已经平滑过了
        # self.smoothed_veh_poses = self._smooth_vehicle_poses()
        self.smoothed_veh_poses = list(self.veh_poses)
        self.smoothed_tracks = {}
        print(
            f"[INFO] 数据准备完成: vehicle pose {len(self.smoothed_veh_poses)} 条, "
            f"object track {len(self.tracks)} 条；object 将在按帧采样后平滑"
        )

    def _smooth_vehicle_poses(self) -> List[VehiclePose]:
        """
        对车辆位姿序列做前向卡尔曼滤波 + RTS 反向平滑.

        状态: [x, y, z, yaw,  vx, vy, vz, vyaw]  (8D)
        """
        if len(self.veh_poses) < 3:
            print("[WARN] 车辆位姿太少, 跳过滤波")
            return list(self.veh_poses)

        pos_dim = 4   # x, y, z, yaw
        n = len(self.veh_poses)

        # ---- 前向滤波 (同时保存 RTS 所需矩阵) ----
        forward_ts:   List[float]       = []
        forward_x:    List[np.ndarray]  = []   # 滤波后状态 x[i]
        forward_P:    List[np.ndarray]  = []   # 滤波后协方差 P[i]
        forward_F:    List[np.ndarray]  = []   # F[i]: t_{i-1}→t_i (i>=1 时有效)
        forward_xpred: List[np.ndarray] = []   # 预测状态 (update 前, i>=1 时有效)
        forward_Ppred: List[np.ndarray] = []   # 预测协方差 (update 前, i>=1 时有效)

        init_meas = np.array([
            self.veh_poses[0].x,
            self.veh_poses[0].y,
            self.veh_poses[0].z,
            self.veh_poses[0].yaw,
        ])
        kf = ConstantVelocityKF(
            pos_dim=pos_dim,
            init_meas=init_meas,
            init_t=self.veh_poses[0].timestamp,
            q_pos=0.05,
            q_vel=0.5,
            r_pos=0.02,
        )

        # 第 0 帧
        forward_ts.append(self.veh_poses[0].timestamp)
        forward_x.append(kf.x.copy())
        forward_P.append(kf.P.copy())
        forward_F.append(np.zeros((kf.dim, kf.dim)))       # 占位
        forward_xpred.append(np.zeros(kf.dim))              # 占位
        forward_Ppred.append(np.zeros((kf.dim, kf.dim)))    # 占位

        for i in range(1, n):
            vp = self.veh_poses[i]
            meas = np.array([vp.x, vp.y, vp.z, vp.yaw])

            est_yaw = kf.x[3]
            meas[3] = est_yaw + _normalize_angle(meas[3] - est_yaw)

            kf.predict(vp.timestamp)
            # 保存预测后的中间矩阵
            forward_F.append(kf.F_saved.copy())
            forward_xpred.append(kf.x_pred.copy())
            forward_Ppred.append(kf.P_pred.copy())

            kf.update(meas)
            kf.x[3] = _normalize_angle(kf.x[3])

            forward_ts.append(vp.timestamp)
            forward_x.append(kf.x.copy())
            forward_P.append(kf.P.copy())

        # ---- RTS 反向平滑 ----
        smoothed_x = [s.copy() for s in forward_x]
        for i in range(n - 2, -1, -1):
            P_i   = forward_P[i]
            F_next = forward_F[i + 1]
            P_pred_next = forward_Ppred[i + 1]
            C = P_i @ F_next.T @ np.linalg.inv(P_pred_next)
            residual = smoothed_x[i + 1] - forward_xpred[i + 1]
            residual[3] = _normalize_angle(residual[3])  # yaw 残差归一化到 [-π, π]
            smoothed_x[i] = forward_x[i] + C @ residual
            smoothed_x[i][3] = _normalize_angle(smoothed_x[i][3])

        # ---- 组装输出 ----
        result: List[VehiclePose] = []
        for i in range(n):
            vp = self.veh_poses[i]
            x = smoothed_x[i]
            result.append(VehiclePose(
                timestamp = forward_ts[i],
                x         = float(x[0]),
                y         = float(x[1]),
                z         = float(x[2]),
                yaw       = float(x[3]),
                pitch     = vp.pitch,
                roll      = vp.roll,
            ))

        return result

    def _smooth_object_tracks(
        self,
        sampled_tracks: Dict[str, List[ObjBbox]],
        sigma_x: float = 4.0,
        sigma_y: float = 4.0,
        sigma_z: float = 2.5,
        sigma_yaw: float = 2.5,
        gaussian_truncate: float = 3.0,
    ) -> Dict[str, List[ObjBbox]]:
        """
        对已经按 data_frame 时间戳采样后的目标序列做高斯局部线性平滑。

        与普通的 ``reflect padding + Gaussian convolution`` 不同，本实现会在
        每一个采样点附近做高斯加权的一阶局部拟合：

            value(offset) = beta_0 + beta_1 * offset

        平滑结果取 beta_0。这样做有两个优点：

        1. 首尾位置只使用真实存在的点，不引入镜像点；
        2. 对匀速直线轨迹，一阶趋势在首尾也能被保留，不会出现端点回折、
           突然贴回原始点或前几帧偏移异常。

        仅平滑 x、y、z、yaw；尺寸保持最近原始观测值。
        sigma 的单位是采样后的输出帧。
        """
        dynamic_types = {"Pedestrian", "Motorcycle", "Bicycle", "Vehicle"}

        def _gaussian_local_linear_smooth(
            values: np.ndarray,
            sigma: float,
        ) -> np.ndarray:
            """
            高斯加权局部线性平滑。

            相比直接卷积，该方法在序列边界不会使用 reflect/mirror padding，
            并且能保持局部线性运动趋势，因此首尾偏差通常更小。
            """
            values = np.asarray(values, dtype=np.float64)
            n = len(values)

            if n <= 2 or sigma <= 0:
                return values.copy()

            radius = max(1, int(round(gaussian_truncate * sigma)))
            result = np.empty_like(values)

            for center in range(n):
                left = max(0, center - radius)
                right = min(n, center + radius + 1)

                indices = np.arange(left, right, dtype=np.float64)
                offsets = indices - float(center)
                local_values = values[left:right]

                weights = np.exp(-0.5 * (offsets / sigma) ** 2)

                # X = [1, offset]，在当前位置 offset=0，预测值就是 beta[0]。
                X = np.column_stack((np.ones_like(offsets), offsets))
                XtW = X.T * weights
                normal_matrix = XtW @ X
                rhs = XtW @ local_values

                # 极短窗口或数值退化时，退化为归一化高斯加权平均。
                try:
                    beta = np.linalg.solve(
                        normal_matrix + np.eye(2) * 1e-12,
                        rhs,
                    )
                    result[center] = beta[0]
                except np.linalg.LinAlgError:
                    result[center] = np.sum(weights * local_values) / np.sum(weights)

            return result

        def _smooth_yaw(
            values: np.ndarray,
            sigma: float,
        ) -> np.ndarray:
            # 先展开角度，避免 pi/-pi 跳变被当成大幅运动。
            values = np.asarray(values, dtype=np.float64)
            unwrapped = np.unwrap(values)
            smoothed = _gaussian_local_linear_smooth(unwrapped, sigma)
            return (smoothed + np.pi) % (2.0 * np.pi) - np.pi

        result: Dict[str, List[ObjBbox]] = {}

        for obj_id, samples in sampled_tracks.items():
            if not samples:
                continue

            samples = sorted(samples, key=lambda box: box.timestamp)
            obj_type = samples[0].obj_type
            n = len(samples)

            # 静态目标或过短序列不做平滑，直接复制。
            if obj_type not in dynamic_types or n < self.MIN_TRACK_LEN:
                result[obj_id] = [
                    ObjBbox(
                        timestamp=box.timestamp,
                        obj_id=box.obj_id,
                        obj_type=box.obj_type,
                        x=box.x,
                        y=box.y,
                        z=box.z,
                        yaw=box.yaw,
                        length=box.length,
                        width=box.width,
                        height=box.height,
                    )
                    for box in samples
                ]
                continue

            raw_x = np.asarray([box.x for box in samples], dtype=np.float64)
            raw_y = np.asarray([box.y for box in samples], dtype=np.float64)
            raw_z = np.asarray([box.z for box in samples], dtype=np.float64)
            raw_yaw = np.asarray([box.yaw for box in samples], dtype=np.float64)

            x_sm = _gaussian_local_linear_smooth(raw_x, sigma_x)
            y_sm = _gaussian_local_linear_smooth(raw_y, sigma_y)
            z_sm = _gaussian_local_linear_smooth(raw_z, sigma_z)
            yaw_sm = _smooth_yaw(raw_yaw, sigma_yaw)

            # 局部线性高斯平滑在边界处只能使用单侧窗口，
            # 因此第 0 帧和最后一帧可能被外推到原始点之外。
            # 这里只固定真正的两个端点，不约束第 1/倒数第 2 帧，
            # 避免之前“连续多帧锚定”造成的端点折线或突变。
            x_sm[0], x_sm[-1] = raw_x[0], raw_x[-1]
            y_sm[0], y_sm[-1] = raw_y[0], raw_y[-1]
            z_sm[0], z_sm[-1] = raw_z[0], raw_z[-1]
            yaw_sm[0], yaw_sm[-1] = raw_yaw[0], raw_yaw[-1]

            result[obj_id] = [
                ObjBbox(
                    timestamp=box.timestamp,
                    obj_id=box.obj_id,
                    obj_type=box.obj_type,
                    x=float(x_sm[i]),
                    y=float(y_sm[i]),
                    z=float(z_sm[i]),
                    yaw=float(yaw_sm[i]),
                    # 尺寸直接使用最近原始观测，不做平均或平滑。
                    length=box.length,
                    width=box.width,
                    height=box.height,
                )
                for i, box in enumerate(samples)
            ]

        return result

    def print_object_xyz_comparison(
        self,
        obj_id: str,
        raw_sampled_tracks: Dict[str, List[ObjBbox]],
        smoothed_tracks: Dict[str, List[ObjBbox]],
        max_points: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        """
        对比按 data_frame 时间采样后、平滑前后的 2D 点位。

        - 不画连线
        - 不画箭头
        - 只显示前 max_points 个采样点
        - 帧序号直接画在圆点中心
        """
        import matplotlib.pyplot as plt

        if obj_id not in raw_sampled_tracks:
            print(f"[Track comparison] 采样结果中不存在 obj_id={obj_id}")
            return

        if obj_id not in smoothed_tracks:
            print(f"[Track comparison] 平滑结果中不存在 obj_id={obj_id}")
            return

        raw_states = raw_sampled_tracks[obj_id]
        smooth_states = smoothed_tracks[obj_id]

        num_vis = min(len(raw_states), len(smooth_states), max_points)
        if num_vis == 0:
            print(f"[Track comparison] obj_id={obj_id} 没有可用数据")
            return

        raw_x = np.asarray([raw_states[i].x for i in range(num_vis)])
        raw_y = np.asarray([raw_states[i].y for i in range(num_vis)])
        smooth_x = np.asarray([smooth_states[i].x for i in range(num_vis)])
        smooth_y = np.asarray([smooth_states[i].y for i in range(num_vis)])

        fig, ax = plt.subplots(figsize=(60, 50))

        ax.scatter(
            raw_x,
            raw_y,
            s=220,
            marker="o",
            alpha=0.45,
            label="sampled raw position",
        )
        ax.scatter(
            smooth_x,
            smooth_y,
            s=140,
            marker="o",
            alpha=0.9,
            label="smoothed position",
        )

        for i in range(num_vis):
            ax.text(
                raw_x[i],
                raw_y[i],
                str(i),
                ha="center",
                va="center",
                fontsize=7,
                alpha=0.75,
            )
            ax.text(
                smooth_x[i],
                smooth_y[i],
                str(i),
                ha="center",
                va="center",
                fontsize=7,
                alpha=0.95,
            )

        ax.set_title(
            f"Sampled vs Smoothed 2D Position: obj_id={obj_id} "
            f"(first {num_vis} points)"
        )
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=180, bbox_inches="tight")
            print(f"[Track comparison] 图片已保存至: {save_path}")

        plt.show()

    def write_labels(self):
        """
        两阶段生成标签：

        第一阶段：
            对每个 data_frame 的 main_timestamp，从每条原始 object track
            中直接选择时间最接近的观测，不做连续插值。

        第二阶段：
            将同一 obj_id 的采样结果按输出帧排序，对 x/y/z/yaw 做高斯平滑，
            再执行 world -> vehicle 坐标变换并写出 JSON。
        """
        dynamic_types = {"Pedestrian", "Motorcycle", "Bicycle", "Vehicle"}

        # 每个输出帧对应的原始采样结果。
        sampled_by_frame: List[List[ObjBbox]] = []
        sampled_tracks: Dict[str, List[ObjBbox]] = defaultdict(list)

        # ========================================================
        # 第一阶段：按 data_frame 时间戳选择最近原始观测
        # ========================================================
        for item in self.data_frame_seq:
            target_ts = float(item["main_timestamp"])
            frame_boxes: List[ObjBbox] = []

            for obj_id, obs_list in self.tracks.items():
                if not obs_list:
                    continue

                # 只在原始 track 的有效时间范围内输出该目标。
                if not (
                    obs_list[0].timestamp
                    <= target_ts
                    <= obs_list[-1].timestamp
                ):
                    continue

                obj_bbox = self._sample_track_at(
                    obs_list=obs_list,
                    target_ts=target_ts,
                )

                frame_boxes.append(obj_bbox)
                sampled_tracks[obj_id].append(obj_bbox)

            sampled_by_frame.append(frame_boxes)

        print(
            f"[INFO] 按输出帧完成最近邻采样: "
            f"{len(sampled_by_frame)} 帧, {len(sampled_tracks)} 条 track"
        )

        # ========================================================
        # 第二阶段：对采样后的序列做高斯平滑
        # ========================================================
        smoothed_tracks = self._smooth_object_tracks(sampled_tracks)
        self.smoothed_tracks = smoothed_tracks

        # 通过 (obj_id, timestamp) 找回每个输出帧对应的平滑结果。
        smoothed_lookup: Dict[Tuple[str, float], ObjBbox] = {}
        for obj_id, boxes in smoothed_tracks.items():
            for box in boxes:
                smoothed_lookup[(obj_id, box.timestamp)] = box

        # 如需调试某条轨迹，可以取消下面注释并修改 obj_id。
        # self.print_object_xyz_comparison(
        #     obj_id="7166",
        #     raw_sampled_tracks=sampled_tracks,
        #     smoothed_tracks=smoothed_tracks,
        #     save_path="smooth.png",
        # )

        # ========================================================
        # 第三阶段：坐标变换并逐帧写出
        # ========================================================
        for frame_idx, item in enumerate(self.data_frame_seq):
            target_ts = float(item["main_timestamp"])
            label_file_name = item["data_frame_path"]
            veh_pose = self._get_veh_pos_at(target_ts)

            target_bboxes: List[ObjBbox] = []

            for raw_bbox in sampled_by_frame[frame_idx]:
                key = (raw_bbox.obj_id, raw_bbox.timestamp)

                # 动态目标使用平滑结果；静态目标保持最近邻采样结果。
                if raw_bbox.obj_type in dynamic_types:
                    world_bbox = smoothed_lookup.get(key, raw_bbox)
                else:
                    world_bbox = raw_bbox

                veh_x, veh_y, veh_z, veh_yaw = world_to_vehicle(
                    obj_x=world_bbox.x,
                    obj_y=world_bbox.y,
                    obj_z=world_bbox.z,
                    obj_yaw=world_bbox.yaw,
                    veh_pose=veh_pose,
                )

                # 创建新对象，不修改缓存中的世界坐标。
                target_bboxes.append(
                    ObjBbox(
                        timestamp=target_ts,
                        obj_id=world_bbox.obj_id,
                        obj_type=world_bbox.obj_type,
                        x=float(veh_x),
                        y=float(veh_y),
                        z=float(veh_z),
                        yaw=float(veh_yaw),
                        length=world_bbox.length,
                        width=world_bbox.width,
                        height=world_bbox.height,
                    )
                )

            psr_list = [
                self._trans_objbbox_to_psrbbox(bbox)
                for bbox in target_bboxes
            ]
            self._write_label_json(label_file_name, psr_list)

    @ staticmethod
    def _lerp(arr, ts_arr, target_ts):
        if target_ts <= ts_arr[0]:
            return arr[0]
        if target_ts >= ts_arr[-1]:
            return arr[-1]
        
        idx = np.searchsorted(ts_arr, target_ts)
        t0, t1 = ts_arr[idx -1], ts_arr[idx]
        alpha = (target_ts - t0) / (t1 - t0)

        return arr[idx -1] + alpha * (arr[idx] - arr[idx - 1])

    def _get_veh_pos_at(self, target_ts):
        ts_arr = [vp.timestamp for vp in self.smoothed_veh_poses]
        x_arr = [vp.x for vp in self.smoothed_veh_poses]
        y_arr = [vp.y for vp in self.smoothed_veh_poses]
        z_arr = [vp.z for vp in self.smoothed_veh_poses]
        yaw_arr = np.unwrap([vp.yaw for vp in self.smoothed_veh_poses])
        pitch_arr = [vp.pitch for vp in self.smoothed_veh_poses]
        roll_arr = [vp.roll for vp in self.smoothed_veh_poses]

        x = self._lerp(x_arr, ts_arr, target_ts)
        y = self._lerp(y_arr, ts_arr, target_ts)
        z = self._lerp(z_arr, ts_arr, target_ts)
        yaw = _normalize_angle(self._lerp(yaw_arr, ts_arr, target_ts))
        pitch = self._lerp(pitch_arr, ts_arr, target_ts)
        roll = self._lerp(roll_arr, ts_arr, target_ts)

        return VehiclePose(timestamp=target_ts,x=x,y=y,z=z,yaw=yaw,pitch=pitch,roll=roll)

    def _sample_track_at(
        self,
        obs_list: List[ObjectObs],
        target_ts: float,
    ) -> ObjBbox:
        """
        从原始 object track 中选择时间戳距离 target_ts 最近的一条观测。

        不再对 x/y/z/yaw 做连续插值，也不再对尺寸求整条轨迹均值。
        x/y/z/yaw/length/width/height 全部来自同一条最近原始观测，
        但输出 timestamp 使用目标 data_frame 的 target_ts。
        """
        if not obs_list:
            raise ValueError("_sample_track_at 收到空的 obs_list")

        ts_arr = np.asarray(
            [obs.timestamp for obs in obs_list],
            dtype=np.float64,
        )

        # searchsorted 后只需要比较左右两个候选，复杂度 O(log N)。
        right = int(np.searchsorted(ts_arr, target_ts, side="left"))

        if right <= 0:
            nearest_idx = 0
        elif right >= len(obs_list):
            nearest_idx = len(obs_list) - 1
        else:
            left = right - 1
            if abs(target_ts - ts_arr[left]) <= abs(ts_arr[right] - target_ts):
                nearest_idx = left
            else:
                nearest_idx = right

        obs = obs_list[nearest_idx]

        return ObjBbox(
            timestamp=float(target_ts),
            obj_id=obs.obj_id,
            obj_type=obs.obj_type,
            x=obs.x,
            y=obs.y,
            z=obs.z,
            yaw=obs.yaw,
            length=obs.length,
            width=obs.width,
            height=obs.height,
        )
    
    @staticmethod
    def _trans_objbbox_to_psrbbox(objbbox:ObjBbox):
        psrbbox = {
            'timestamp': objbbox.timestamp,
            'obj_id': objbbox.obj_id,
            'obj_type': objbbox.obj_type,
            'psr': {
                'position': {
                    'x': objbbox.x,
                    'y': objbbox.y,
                    'z': objbbox.z,
                },
                'rotation': {
                    'x':0,
                    'y':0,
                    'z':objbbox.yaw
                },
                'scale': {
                    'x':objbbox.length,
                    'y':objbbox.width,
                    'z':objbbox.height
                }
            }
        }
        return psrbbox
    
    def _write_label_json(self, file_name, psr_list):
        label_json_path = self.label_dir / f"{file_name}.json"
        # 写入json
        with open(label_json_path, 'w', encoding='utf-8') as f:
            json.dump(psr_list, f, indent=4)


def main():

    parser = argparse.ArgumentParser(description='从 object_proto.json 中提取 3dbox 信息（带滤波平滑）')
    parser.add_argument('--parsed_data_root', default="home/data/parsed/", help='parsed 数据根目录')
    parser.add_argument('--scene_id', type=str, help='场景 ID')
    parser.add_argument('--time_range', '-t', nargs=2, type=int, metavar=('START', 'END'),
                        required=False, help='切片时间范围, 如 -t 30 60')
    args = parser.parse_args()

    extractor = BoxExtractor(
        parsed_data_root = args.parsed_data_root,
        scene_id         = args.scene_id,
        time_range       = args.time_range,
    )
    extractor.load()
    extractor.smooth()
    extractor.write_labels()


if __name__ == "__main__":
    main()
