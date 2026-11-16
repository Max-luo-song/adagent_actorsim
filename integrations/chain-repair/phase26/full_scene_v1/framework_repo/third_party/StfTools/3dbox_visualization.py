#!/usr/bin/env python3
"""
将3D标注框反投影到相机图像上。

读取label目录下的3D box标注JSON，结合camera_params.json（相机内外参）和
每个时间戳文件夹下的data_frame.json（车辆位姿），将3D框投影到各相机图像上。

用法:
    python project_3dbox_to_images.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--max_frames N]
"""

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


import argparse
import json
import sys
pass  # Path is provided by the portable boundary above
from typing import Dict, Tuple, Optional

import numpy as np

import cv2
from moviepy.editor import VideoFileClip, clips_array

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CAMERA_ID_TO_NAME = {
    75: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
    76: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    77: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    97: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    78: "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    83: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    84: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    79: "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    85: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    86: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    82: "CAM_PBQ_REAR_RESET_OPTICAL_H50",
}

GRID_VIDEO_CAM = [
            "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
            "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
            "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
            "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
            "CAM_PBQ_REAR_RESET_OPTICAL_H50",
            "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
            ]

CAM_NUM_TO_NAME = {
    '000.png':'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',
    '001.png':'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60',
    '002.png':'CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30',
    '003.png':'CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15',

    '005.png':'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',
    '006.png':'CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99',
    '007.png':'CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30',

    '009.png':'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',
    '010.png':'CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99',
    '011.png':'CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30',

    '012.png':'CAM_PBQ_REAR_RESET_OPTICAL_H50'
}

# ============================================================
# 绘制
# ============================================================

# 不同目标类型的颜色 (BGR)
TYPE_COLORS = {
    'Vehicle': (0, 255, 0),          # 绿色
    'Car': (0, 255, 0),              # 绿色
    'Pedestrian': (0, 0, 255),       # 红色
    'Motorcycle': (255, 0, 0),       # 蓝色
    'Bicycle': (255, 255, 0),        # 青色
    'TrafficBarrier': (0, 255, 255), # 黄色
}

BEV_FRAME_ALPHA = 0.4       # BEV 框填充透明度
BEV_FRAME_EDGE_COLOR = 'lime'
BEV_ARROW_COLOR = 'yellow'
BEV_DOT_COLOR = 'red'

# 3D框的边定义 (按角点索引)
# 底部: 0-1-2-3, 顶部: 4-5-6-7, 竖线: (0,4), (1,5), (2,6), (3,7)
BOTTOM_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
TOP_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]
VERTICAL_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

# ============================================================
# 坐标系约定
# ============================================================
#
# 世界坐标系 (World): 与车辆系原点不同但轴线对齐
# 车辆坐标系 (Vehicle): X-前, Y-左, Z-上
# 相机坐标系 (Camera):   X-右, Y-下, Z-前 (光轴，OpenCV/ROS标准)
#
# 相机 → 车辆的基变换 (当相机"正装"看向车辆前方时):
#   Cam X (右)  → Veh -Y (右 = -左)
#   Cam Y (下)  → Veh -Z (下 = -上)
#   Cam Z (前)  → Veh  X
R_BASE_CAM2VEH = np.array([
    [0,  0,  1],   # Cam X → Veh
    [-1, 0,  0],   # Cam Y → Veh
    [0, -1,  0],   # Cam Z → Veh
])
R_BASE_VEH2CAM = R_BASE_CAM2VEH.T  # = [[0,-1,0],[0,0,-1],[1,0,0]]

# ============================================================
# 坐标系变换 (全部使用 4×4 齐次变换矩阵)
# ============================================================


# ==================工具函数===================
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Euler角 → 3×3旋转矩阵 (绕固定轴: 先Z, 再Y, 再X)。
    用于绕**车辆坐标系**轴的旋转。
    旋转顺序: R = Rz(yaw) · Ry(pitch) · Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


def make_transform_4x4(R, t):
    """由 3×3 旋转矩阵 R 和 3×1 平移向量 t 构造 4×4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _transform_points(points, T):
    """
    用 4×4 齐次变换矩阵 T 批量变换点集。

    Args:
        points: (N, 3) ndarray
        T: (4, 4) 齐次变换矩阵

    Returns:
        (N, 3) 变换后的点
    """
    N = points.shape[0]
    points_h = np.column_stack([points, np.ones(N, dtype=np.float64)])  # (N, 4)
    return (T @ points_h.T).T[:, :3]  # (N, 3)


class BboxProjector():

    def __init__(self, data_dir, output_folder, img_input_dir=None):

        # ============设置路径=============
        self.data_dir = Path(data_dir)
        self.label_dir = self.data_dir / 'label_pred'
        self.camera_params_path = self.data_dir / 'camera_params.json'
        self.data_frame_seq_path = self.data_dir / 'data_frame_seq.json'
        self.output_dir = self.data_dir / output_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_input_dir = Path(img_input_dir) if img_input_dir else None

        # ============加载数据=============
        self.camera_params = {}
        self.timestamp_folders = []
    
    def load(self):
        self._load_cam_params()
        self._load_timestamp()

    def _load_cam_params(self):
        if not self.camera_params_path.exists():
            print(f"[ERROR] 找不到相机参数文件: {self.camera_params_path}")
            sys.exit(1)
        with open(self.camera_params_path) as f:
            all_cam_params = _hugsim_expand(json.load(f))

        self.camera_params = {
            cam_name: params
            for cam_name, params in all_cam_params.items()
            if cam_name in CAMERA_ID_TO_NAME.values()
        }
        print(f"[INFO] 加载了 {len(self.camera_params)} 个相机的参数")
    
    def _load_timestamp(self):
        if not self.data_frame_seq_path.exists():
            print(f"[ERROR] 找不到时间戳记录文件: {self.data_frame_seq_path}")
            sys.exit(1)
        with open (self.data_frame_seq_path) as f:
            data_frame_seq_items = _hugsim_expand(json.load(f))["data_frame_seq_items"]
        self.timestamp_folders = [d["data_frame_path"] for d in data_frame_seq_items]

        print(f"[INFO] 有 {len(self.timestamp_folders)} 帧数据待处理")
    
    def project(self):

        total_boxes_drawn = 0
        total_images = 0

        for ts in self.timestamp_folders:
            # 标签json路径和图片路径(都和timestamp有关)
            json_name = str(ts) + '.json'
            label_path = self.label_dir / json_name
            images_dir = self.data_dir / ts

            # 从.json中加载label
            if not label_path.exists():
                continue
            with open(label_path) as f:
                labels = _hugsim_expand(json.load(f))
            if not labels:
                continue
            
            # 图片路径
            image_files = sorted([
                f for f in images_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])

            for img_path in image_files:
                cam_name = self._extract_camera_name(img_path.name, self.camera_params.keys())
                if cam_name is None:
                    continue
                if cam_name not in self.camera_params:
                    continue
                    
                # 读取图片对应相机内外参
                cam_param = self.camera_params[cam_name]
                extrinsics = cam_param['camera_to_vehicle_extrinsics']
                intrinsics = cam_param['intrinsics'].copy()  # 后面要修改intrinsics

                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # =============投影每个3d框===============
                for obj in labels:
                    psr = obj['psr']
                    position = psr['position']
                    rotation = psr['rotation']
                    scale = psr['scale']
                    obj_type = obj.get('obj_type', 'unknown')
                    obj_id = obj.get('obj_id', '')

                    try:
                        # 1) 局部系下8个角点
                        corners = self._get_8_corners(position, rotation, scale)

                        # 2) 车辆 → 相机 (3D)
                        corners_camera = self._vehicle_to_camera(corners, extrinsics)

                        # 3) 近平面裁剪 + 相机 → 像素
                        edges_2d = self._clip_and_project_edges(corners_camera, intrinsics)

                        # 4) 绘制
                        if edges_2d:
                            self._draw_3dbox_on_image(image, edges_2d, obj_type, obj_id)
                            total_boxes_drawn += 1
                    except Exception as e:
                        print(f"    [WARN] 投影框失败 (obj={obj_id}): {e}")
                
                # 保存标注后的图像（按相机名称归入子文件夹）
                cam_out_dir = self.output_dir / cam_name
                cam_out_dir.mkdir(parents=True, exist_ok=True)
                img_save_name = f"{img_path.name}"
                img_save_path = cam_out_dir / img_save_name
                cv2.imwrite(str(img_save_path), image)
                total_images += 1
            
            print(f"  ✓ {ts}: {len(image_files)}张图, {len(labels)}个框")

        print(f"\n{'='*60}")
        print(f"[DONE] 完成!")
        print(f"  处理了 {total_images} 张图像")
        print(f"  绘制了 {total_boxes_drawn} 个3D框")

        print(f"  结果保存到: {self.output_dir}")

    def project_flat(self):
        """
        从扁平图片目录读取图像并投影 3D 框。

        图片命名: {frame_idx:06d}_{cam_num:03d}.png
        例: 000000_006.png → frame_idx=0, cam_num=6 → 查询 CAM_NUM_TO_NAME 得相机名

        标签仍然从 data_dir/label_pred/{timestamp}.json 读取，
        timestamp 按 data_frame_seq.json 的顺序与 frame_idx 一一对应。
        """
        if self.img_input_dir is None:
            print("[ERROR] project_flat 需要 --img_input_dir 参数")
            sys.exit(1)
        if not self.img_input_dir.exists():
            print(f"[ERROR] 图片目录不存在: {self.img_input_dir}")
            sys.exit(1)

        # ---- 建立 cam_num → cam_name 映射 ----
        # CAM_NUM_TO_NAME 的 key 如 '006.png', 提取数字部分
        num_to_name: Dict[int, str] = {}
        for key, cam_name in CAM_NUM_TO_NAME.items():
            try:
                num_str = key.replace('.png', '').replace('.jpg', '')
                num_to_name[int(num_str)] = cam_name
            except ValueError:
                pass
        print(f"[INFO] 相机编号映射: {sorted(num_to_name.items())}")

        # ---- 预扫描图片目录, 建立 (frame_idx, cam_num) → path 索引 ----
        print(f"[INFO] 扫描图片目录: {self.img_input_dir}")
        img_index: Dict[Tuple[int, int], Path] = {}
        for f in self.img_input_dir.iterdir():
            if f.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            parsed = self._parse_flat_image_name(f.name)
            if parsed is None:
                continue
            frame_idx, cam_num = parsed
            img_index[(frame_idx, cam_num)] = f
        n_frames = max(k[0] for k in img_index) + 1 if img_index else 0
        print(f"[INFO] 找到 {len(img_index)} 张图片, 共 {n_frames} 帧, "
              f"{len(set(k[1] for k in img_index))} 个相机")

        total_boxes_drawn = 0
        total_images = 0

        for frame_idx, ts in enumerate(self.timestamp_folders):
            # ---- 加载标签 ----
            json_name = f"{ts}.json"
            label_path = self.label_dir / json_name
            if not label_path.exists():
                continue
            with open(label_path) as f:
                labels = _hugsim_expand(json.load(f))
            if not labels:
                continue

            # ---- 遍历每个相机 ----
            for cam_num, cam_name in sorted(num_to_name.items()):
                if cam_name not in self.camera_params:
                    continue
                img_key = (frame_idx, cam_num)
                if img_key not in img_index:
                    continue

                img_path = img_index[img_key]
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                cam_param = self.camera_params[cam_name]
                extrinsics = cam_param['camera_to_vehicle_extrinsics']
                intrinsics = cam_param['intrinsics'].copy()

                # ---- 投影每个 3D 框 ----
                for obj in labels:
                    psr = obj['psr']
                    position = psr['position']
                    rotation = psr['rotation']
                    scale = psr['scale']
                    obj_type = obj.get('obj_type', 'unknown')
                    obj_id = obj.get('obj_id', '')

                    try:
                        corners = self._get_8_corners(position, rotation, scale)
                        corners_camera = self._vehicle_to_camera(corners, extrinsics)
                        edges_2d = self._clip_and_project_edges(corners_camera, intrinsics)
                        if edges_2d:
                            self._draw_3dbox_on_image(image, edges_2d, obj_type, obj_id)
                            total_boxes_drawn += 1
                    except Exception as e:
                        print(f"    [WARN] 投影框失败 (frame={frame_idx} cam={cam_num} obj={obj_id}): {e}")

                # ---- 保存 ----
                output_dir = self.data_dir / "result_visual"
                cam_out_dir = output_dir / cam_name
                cam_out_dir.mkdir(parents=True, exist_ok=True)
                img_save_name = f"{frame_idx:06d}_{cam_num:03d}.png"
                cv2.imwrite(str(cam_out_dir / img_save_name), image)
                total_images += 1

            if (frame_idx + 1) % 5 == 0 or frame_idx == 0:
                print(f"  进度: {frame_idx + 1}/{len(self.timestamp_folders)}", end='\r')

        print(f"\n{'='*60}")
        print(f"[DONE] 完成!")
        print(f"  处理了 {total_images} 张图像")
        print(f"  绘制了 {total_boxes_drawn} 个3D框")
        print(f"  结果保存到: {self.output_dir}")

    @staticmethod
    def _parse_flat_image_name(filename: str) -> Optional[Tuple[int, int]]:
        """
        解析扁平目录下的图片文件名。

        格式: {frame_idx:06d}_{cam_num:03d}.png → (frame_idx, cam_num)
        例: 000000_006.png → (0, 6)
            000123_010.png → (123, 10)
        """
        stem = Path(filename).stem          # "000000_006"
        parts = stem.split('_')
        if len(parts) != 2:
            return None
        try:
            frame_idx = int(parts[0])
            cam_num = int(parts[1])
            return frame_idx, cam_num
        except ValueError:
            return None

    @staticmethod
    def _extract_camera_name(filename, camera_list):
        """
        从图片文件名中提取相机名称。
        文件名格式: {CAMERA_NAME}-{timestamp}.jpg

        通过匹配known_cameras中的最长匹配前缀来确定相机名。
        """
        name = Path(filename).stem
        # 按长度降序排列，优先匹配更长的名称（避免 "CAM_PBQ_FRONT" 匹配到 "CAM_PBQ_FRONT_WIDE" 的部分）
        for cam_name in sorted(camera_list, key=len, reverse=True):
            if name.startswith(cam_name):
                return cam_name
        return None

    @staticmethod
    def _get_8_corners(position, rotation, scale):
        """
        获取3D框的8个角点坐标。

        局部坐标系: x-前(长), y-左(宽), z-上(高), 原点在框的几何中心

        Args:
            position: {x, y, z}  框的位置
            rotation: {x, y, z}  旋转 (roll, pitch, yaw, 弧度)
            scale: {x, y, z}     x=长, y=宽, z=高

        Returns:
            corners: (8, 3) ndarray, 8个角点坐标
        """
        l, w, h = scale['x'], scale['y'], scale['z']
        tx, ty, tz = position['x'], position['y'], position['z']


        # 局部坐标系下的8个角点 (中心在原点)
        corners_local = np.array([
            [+l / 2, +w / 2, -h / 2],  # 0: 前-左-下
            [+l / 2, -w / 2, -h / 2],  # 1: 前-右-下
            [-l / 2, -w / 2, -h / 2],  # 2: 后-右-下
            [-l / 2, +w / 2, -h / 2],  # 3: 后-左-下
            [+l / 2, +w / 2, +h / 2],  # 4: 前-左-上
            [+l / 2, -w / 2, +h / 2],  # 5: 前-右-上
            [-l / 2, -w / 2, +h / 2],  # 6: 后-右-上
            [-l / 2, +w / 2, +h / 2],  # 7: 后-左-上
        ], dtype=np.float64)

        # 绕Z,Y,X旋转 (yaw/pitch/roll → Rz·Ry·Rx)，构成 4×4 齐次变换
        roll_x, pitch_y, yaw_z = rotation['x'], rotation['y'], rotation['z']
        R = euler_to_rotation_matrix(roll_x, pitch_y, yaw_z)
        T_box = make_transform_4x4(R, np.array([tx, ty, tz]))

        return _transform_points(corners_local, T_box)
    
    @staticmethod
    def _vehicle_to_camera(points_vehicle, extrinsics):
        """
        车辆坐标系 → 相机坐标系。

        extrinsics (camera_to_vehicle) 定义了相机在车辆系下的完整位姿:
            X_veh = R_ypr · R_BASE_CAM2VEH · X_cam + t
        其中 R_ypr = Rz(yaw)·Ry(pitch)·Rx(roll) 绕**车辆系**轴的额外旋转。

        逆变换 (车辆→相机):
            T_camera←vehicle = [R_BASE_VEH2CAM·R_ypr^T | -R_BASE_VEH2CAM·R_ypr^T·t]

        Args:
            points_vehicle: (N, 3) 车辆系点
            extrinsics: {x, y, z, roll, pitch, yaw}

        Returns:
            (N, 3) 相机系点
        """
        t = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])
        R_ypr = euler_to_rotation_matrix(extrinsics['roll'], extrinsics['pitch'], extrinsics['yaw'])
        R_veh2cam = R_BASE_VEH2CAM @ R_ypr.T
        T_v2c = make_transform_4x4(R_veh2cam, -R_veh2cam @ t)
        return _transform_points(points_vehicle, T_v2c)

    @staticmethod
    def _clip_and_project_edges(corners_camera, intrinsics):
        """
        在 3D 相机空间对 8 角点 box 做近平面裁剪，然后投影到像素坐标。

        对每条边: 两个端点都在相机前方 → 正常投影
                  一个在前一个在后 → 在近平面处插值截断
                  都在后方 → 丢弃

        Args:
            corners_camera: (8, 3) 相机坐标系下的 8 个角点 (z 轴朝前)
            intrinsics: 相机内参 dict

        Returns:
            list of (pt1_2d, pt2_2d), 每条线段两个端点的像素坐标
        """
        NEAR = 0.5  # 近平面距离 (米), 太小时投影坐标会爆炸
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        def _project_one(pt):
            """单点 3D → 2D 像素 (针孔模型)"""
            u = fx * pt[0] / pt[2] + cx
            v = fy * pt[1] / pt[2] + cy
            return np.array([u, v])

        all_edges = BOTTOM_EDGES + TOP_EDGES + VERTICAL_EDGES
        segments = []

        for i1, i2 in all_edges:
            p1 = corners_camera[i1]
            p2 = corners_camera[i2]
            z1, z2 = p1[2], p2[2]

            if z1 > NEAR and z2 > NEAR:
                # 整条边在相机前方 → 直接投影
                segments.append((_project_one(p1), _project_one(p2)))
            elif z1 > NEAR or z2 > NEAR:
                # 单点在前 → 在近平面处插值截断
                # p(t) = p1 + t * (p2 - p1), 令 p(t).z = NEAR
                t = (NEAR - z1) / (z2 - z1)
                t = np.clip(t, 0.0, 1.0)
                p_clip = p1 + t * (p2 - p1)
                if z1 > NEAR:
                    segments.append((_project_one(p1), _project_one(p_clip)))
                else:
                    segments.append((_project_one(p_clip), _project_one(p2)))
            # else: 两个点都在近平面后方 → 丢弃

        return segments if segments else None

    @staticmethod
    def _draw_3dbox_on_image(image, edges_2d, obj_type='Vehicle', obj_id=''):
        """
        在图像上绘制 3D 框（已裁剪好的线段）。

        Args:
            image: BGR图像 (会被原地修改)
            edges_2d: list of (pt1, pt2), 每条线段两个端点的像素坐标
            obj_type: 目标类型
            obj_id: 目标ID
        """
        h_img, w_img = image.shape[:2]
        color = TYPE_COLORS.get(obj_type, (255, 255, 255))

        # 收集所有有效端点用于视锥体判断和标签位置
        all_pts = []
        for pt1, pt2 in edges_2d:
            if np.all(np.isfinite(pt1)) and np.all(np.abs(pt1) < max(w_img, h_img) * 20):
                all_pts.append(pt1)
            if np.all(np.isfinite(pt2)) and np.all(np.abs(pt2) < max(w_img, h_img) * 20):
                all_pts.append(pt2)

        if len(all_pts) < 2:
            return

        all_pts = np.array(all_pts)
        u_min, v_min = all_pts.min(axis=0)
        u_max, v_max = all_pts.max(axis=0)
        # 视锥体裁剪
        if u_max < 0 or u_min > w_img or v_max < 0 or v_min > h_img:
            return

        # 画每条边 (cv2.line 会自动裁剪越界坐标)
        for pt1, pt2 in edges_2d:
            if not (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2))):
                continue
            cv2.line(image, tuple(pt1.astype(np.int32)), tuple(pt2.astype(np.int32)),
                     color, 1)

        # 标签: 用所有端点的均值位置
        center = np.mean(all_pts, axis=0).astype(np.int32)
        label = f"{obj_type[:3]}_{obj_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(image,
                    (center[0] - tw // 2 - 2, center[1] - th - 4),
                    (center[0] + tw // 2 + 2, center[1]),
                    color, -1)
        cv2.putText(image, label, (center[0] - tw // 2, center[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    def visualize(self, bev_video_path=None):
        """
        各相机图片 → 单相机视频 → 6 相机网格视频, 可选与 BEV 视频左右拼接.

        Args:
            bev_video_path: BEV 视频路径 (Path or str), 为 None 则跳过拼接
        """
        video_paths = []

        # ---- 1. 每个相机图片 → 视频 ----
        for cam_name in GRID_VIDEO_CAM:
            cam_img_dir = self.output_dir / cam_name

            image_files = sorted([
                f for f in cam_img_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            output_name = str(cam_name) + ".mp4"

            self._images_to_video(images=image_files,
                                  output=self.output_dir / output_name,
                                  fps=10.0)

            if cam_name in GRID_VIDEO_CAM:
                 video_paths.append(self.output_dir / output_name)

        # ---- 2. 6 相机网格视频 ----
        grid_path = self.output_dir / 'grid_video.mp4'
        self._create_video_grid(video_paths, rows=2, cols=3,
                                output=grid_path, fps=10.0)

        # ---- 3. 与 BEV 视频左右拼接 ----
        if bev_video_path is not None and Path(bev_video_path).exists():
            print("[INFO] 拼接 grid + BEV 左右视频...")
            grid_clip = VideoFileClip(str(grid_path))
            bev_clip  = VideoFileClip(str(bev_video_path))

            # 统一高度 (以 grid 为准)
            target_h = grid_clip.h
            if bev_clip.h != target_h:
                bev_clip = bev_clip.resize(height=target_h)

            final = clips_array([[grid_clip, bev_clip]])
            final_path = self.output_dir / 'final_video.mp4'
            final.to_videofile(str(final_path), fps=10.0)

            grid_clip.close()
            bev_clip.close()
            final.close()
            print(f"[INFO] 最终视频已保存至: {final_path}")

    def _images_to_video(self,
                        images: list[Path],
                        output: str = "output.mp4",
                        fps: float = 10.0,
                        width: int | None = None,
                        height: int | None = None,
                        codec: str = "mp4v",
                        keep_aspect: bool = True,
                        verbose: bool = True,
                    ) -> str:
        """
        将图片列表拼接为一个视频。

        参数:
            images: 图片文件路径列表
            output: 输出视频文件路径
            fps: 帧率
            width, height: 强制输出分辨率（None 则取首图尺寸）
            codec: FourCC 编码器（mp4v → .mp4, xvid → .avi, avc1 → .mp4 H.264）
            keep_aspect: 缩放时是否保持宽高比（letterbox）
            verbose: 是否打印进度
        """
        if not images:
            raise ValueError("图片列表为空")

        # 读取首图，确定视频尺寸
        first = cv2.imread(str(images[0]))
        if first is None:
            raise ValueError(f"无法读取图片: {images[0]}")

        h, w = first.shape[:2]

        if width is None and height is None:
            width, height = w, h
        elif width is None:
            width = int(height * w / h)
        elif height is None:
            height = int(width * h / w)

        target_size = (width, height)

        if verbose:
            print(f"输入尺寸: {w}x{h}")
            print(f"输出尺寸: {width}x{height}")
            print(f"图片数量: {len(images)}")
            print(f"帧率:     {fps} fps")
            print(f"时长:     {len(images) / fps:.1f} 秒")
            print(f"编码器:   {codec}")
            print(f"输出文件: {output}")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output, fourcc, fps, target_size)

        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器（编码器 {codec} 可能不支持此扩展名）")

        total = len(images)
        for i, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"⚠ 跳过损坏图片: {img_path}", file=sys.stderr)
                continue

            # 缩放到目标尺寸
            if (frame.shape[1], frame.shape[0]) != target_size:
                if keep_aspect:
                    frame = self.__letterbox_resize(frame, width, height)
                else:
                    frame = cv2.resize(frame, target_size)

            writer.write(frame)

            if verbose and (i + 1) % max(1, total // 20) == 0:
                pct = (i + 1) / total * 100
                print(f"\r进度: {i + 1}/{total} ({pct:.0f}%)", end="", flush=True)

        writer.release()
        if verbose:
            print(f"\r完成! {total} 张图片已写入 {output}")

        return output

    def __letterbox_resize(img, target_w: int, target_h: int) -> "cv2.Mat":
        """保持宽高比缩放到目标尺寸，不足部分用黑色填充（letterbox/pillarbox）。"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = (0, 0, 0)  # 黑色背景
        if len(img.shape) == 3:
            canvas = (0, 0, 0)
        result = cv2.copyMakeBorder(
            resized,
            top=(target_h - new_h) // 2,
            bottom=target_h - new_h - (target_h - new_h) // 2,
            left=(target_w - new_w) // 2,
            right=target_w - new_w - (target_w - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=canvas,
        )
        return result
        # 视频拼接

        # bev2d框

    @staticmethod
    def _create_video_grid(video_paths, rows, cols, output, fps):
        """
        将多个视频拼成 rows 行 cols 列的网格
        
        参数:
            video_paths: list, 视频文件路径列表（按行排列，从左到右）
            rows: int, 行数
            cols: int, 列数
            output_path: str, 输出文件路径
            fps: int, 输出视频的帧率
        """
        total_needed = rows * cols
        if len(video_paths) != total_needed:
            raise ValueError(f"需要恰好 {total_needed} 个视频，但传入了 {len(video_paths)} 个")
        
        # 加载所有视频
        clips = [VideoFileClip(str(path)) for path in video_paths]

        # 构建二维列表
        grid = []
        for i in range(rows):
            row_start = i * cols
            grid.append(clips[row_start : row_start + cols])

        # 拼接
        final_clip = clips_array(grid)
        # 导出
        final_clip.to_videofile(str(output), fps=fps)

        print(f"视频已保存至: {output}")
        
        # 释放资源
        for clip in clips:
            clip.close()
        final_clip.close()

    @staticmethod
    def _draw_bev_frame(ax, labels, max_range=50):
        """
        在 matplotlib axes 上绘制单帧 BEV 视图: 同心圆网格 + 2D 目标框。

        Args:
            ax: matplotlib axes
            labels: list of dict, label JSON 中的对象列表
            max_range: 最大显示半径 (m), 默认 50
        """
        ax.clear()
        ax.set_facecolor('black')

        # ---- 1. 同心圆网格 (每 10m) ----
        for r in range(10, max_range + 1, 10):
            circle = plt.Circle((0, 0), r, color='white', linewidth=0.5,
                                fill=False, alpha=0.6)
            ax.add_patch(circle)

        # ---- 2. 坐标范围 & 比例 ----
        margin = max_range + 5
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect('equal')
        ax.axis('off')

        # ---- 3. 绘制每个目标的 2D 框 ----
        for obj in labels:
            psr = obj['psr']
            position = psr['position']
            rotation = psr['rotation']
            scale = psr['scale']

            x   = position['x']
            y   = position['y']
            yaw = rotation['z']
            length = scale['x']
            width  = scale['y']

            # 车辆系 (X-前, Y-左) → 绘图系 (u-右, v-上)
            u_center = -y
            v_center =  x

            # 局部角点 (中心原点, x-前, y-左)
            corners = np.array([
                [-length/2, -width/2],
                [ length/2, -width/2],
                [ length/2,  width/2],
                [-length/2,  width/2],
            ])
            rot_z = np.array([[np.cos(yaw), -np.sin(yaw)],
                              [np.sin(yaw),  np.cos(yaw)]])
            corners_rot = corners @ rot_z.T

            # 变换到绘图坐标
            corners_plot = np.zeros_like(corners_rot)
            corners_plot[:, 0] = -corners_rot[:, 1] + u_center
            corners_plot[:, 1] =  corners_rot[:, 0] + v_center

            # 半透明填充矩形
            poly = patches.Polygon(corners_plot, closed=True,
                                   color=BEV_FRAME_EDGE_COLOR,
                                   alpha=BEV_FRAME_ALPHA, linewidth=1.0)
            ax.add_patch(poly)

            # 中心点
            ax.plot(u_center, v_center, 'o', color=BEV_DOT_COLOR, markersize=2)

            # 朝向箭头
            arrow_len = length / 2
            arrow_end_u = - (y + np.sin(yaw) * arrow_len)
            arrow_end_v =   x + np.cos(yaw) * arrow_len
            ax.arrow(u_center, v_center,
                     arrow_end_u - u_center, arrow_end_v - v_center,
                     head_width=0.5, head_length=0.8,
                     fc=BEV_ARROW_COLOR, ec=BEV_ARROW_COLOR, alpha=0.9)

    def bev(self):
        """
        生成 BEV 2D 框可视化视频, 并与 6 相机拼接视频左右组合为最终输出.
        """
        bev_frames_dir = self.output_dir / 'bev_frames'
        bev_frames_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        frame_paths = []

        for ts in self.timestamp_folders:
            label_path = self.label_dir / f"{ts}.json"
            labels = []
            if label_path.exists():
                with open(label_path) as f:
                    labels = _hugsim_expand(json.load(f))

            self._draw_bev_frame(ax, labels)

            frame_path = bev_frames_dir / f"{ts}.png"
            fig.savefig(str(frame_path), dpi=80, facecolor='black', pad_inches=0)
            frame_paths.append(frame_path)

        plt.close(fig)

        if not frame_paths:
            print("[WARN] BEV: 没有生成任何帧")

        # 生成 BEV 视频
        return self._images_to_video(
            images=sorted(bev_frames_dir.iterdir()),
            output=self.output_dir / 'bev_video.mp4',
            fps=10.0,
        )

def main():
    parser = argparse.ArgumentParser(description='将3D标注框反投影到相机图像上')
    parser.add_argument('--data_dir', type=str,
                        default='/home/chery/Projects/Data/20260526_145008_QCJPSD851972/20260526_145008_QCJPSD851972_3096_3121/',
                        help='解析好的数据文件夹')
    parser.add_argument('--output_folder', type=str, default='visualization',
                        help='输出目录 (默认: data_dir/visualization)')
    parser.add_argument('--img_input_dir', type=str, default=None,
                        help='扁平图片目录 (用于重建结果投影). '
                             '图片命名: {frame_idx:06d}_{cam_num:03d}.png, '
                             '如 /path/to/full_set_40000_gt_rgbs')

    args = parser.parse_args()

    projector = BboxProjector(data_dir=args.data_dir,
                              output_folder=args.output_folder,
                              img_input_dir=args.img_input_dir)

    projector.load()

    if args.img_input_dir:
        projector.project_flat()
    else:
        projector.project()

    bev_path = projector.bev()

    projector.visualize(bev_video_path=bev_path)


if __name__ == '__main__':
    main()

