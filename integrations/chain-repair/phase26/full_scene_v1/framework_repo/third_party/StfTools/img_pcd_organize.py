#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import json
import os
import re
import shutil
import argparse

pass  # Path is provided by the portable boundary above
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

camera_names = [
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    "CAM_PBQ_REAR_RESET_OPTICAL_H50",
]

required_files_list = [
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    "CAM_PBQ_REAR_RESET_OPTICAL_H50",
    "LDR_FRONT",
    "data_frame.json"
]

# ====================图片组织模块=======================
def organize_images_by_majority_decimal3(source_dir, dest_dir=None):
    """
    按小数点后一位分组，文件夹以组内出现最多的后三位命名

    输入布局：图片解析后按相机名存放在 <source_dir>/<camera_name>/*.jpg
              （兼容旧的平铺布局 <source_dir>/*.jpg）

    规则：
    - 小数点后一位相同 → 归入同一文件夹
    - 文件夹名称：取组内出现次数最多的后三位（如 1774424231.827）

    return:
    - 三位时间戳列表,支持给write_data_frame_json和write_data_frame_seq_json使用
    """
    source_path = Path(source_dir)
    if dest_dir is None:
        dest_path = source_path
    else:
        dest_path = Path(dest_dir)
    
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 正则匹配时间戳
    pattern = re.compile(r'.*-(\d+\.\d+)\.jpg$')

    # 第一层分组：按小数点后一位
    groups_by_decimal1 = defaultdict(list)  # {decimal1_key: [(full_timestamp, file_path), ...]}

    # 收集所有文件：图片按相机名分子文件夹存放(<camera_name>/*.jpg)，兼容旧的平铺布局(*.jpg)
    jpg_files = []
    seen_names = set()
    for file_path in sorted(source_path.glob("*/*.jpg")) + sorted(source_path.glob("*.jpg")):
        if file_path.name in seen_names:
            continue
        seen_names.add(file_path.name)
        jpg_files.append(file_path)

    for file_path in jpg_files:
        match = pattern.search(file_path.name)
        if not match:
            print(f"⚠️ 跳过（格式不匹配）: {file_path.name}")
            continue
        
        full_timestamp = match.group(1)  # 如 "1774424231.8278"
        
        # 提取小数点后一位作为分组 key
        if '.' in full_timestamp:
            int_part, dec_part = full_timestamp.split('.')
            dec_1 = dec_part[0] if len(dec_part) > 0 else '0'
            decimal1_key = f"{int_part}.{dec_1}"
        else:
            decimal1_key = f"{full_timestamp}.0"
        
        groups_by_decimal1[decimal1_key].append((full_timestamp, file_path))
    
    print(f"找到 {sum(len(v) for v in groups_by_decimal1.values())} 个图片文件")
    print(f"按小数点后一位分为 {len(groups_by_decimal1)} 组\n")
    
    total_moved = 0
    data_frame_list = []
    
    # 处理每个分组
    print(f"📁开始处理文件分组")
    for decimal1_key, items in groups_by_decimal1.items():
        # print(f"📁 处理分组: {decimal1_key}* ({len(items)} 个文件)")
        
        # 统计该组内所有后三位的出现频率
        decimal3_counter = Counter()
        decimal3_map = defaultdict(list)  # {decimal3_key: [file_path, ...]}
        
        for full_timestamp, file_path in items:
            if '.' in full_timestamp:
                int_part, dec_part = full_timestamp.split('.')
                # 取后三位（不足补0）
                dec_part_3 = (dec_part + '000')[:3]
                decimal3_key = f"{int_part}.{dec_part_3}"
            else:
                decimal3_key = f"{full_timestamp}.000"
            
            decimal3_counter[decimal3_key] += 1
            decimal3_map[decimal3_key].append((full_timestamp, file_path))
        
        # 找出出现次数最多的后三位
        if decimal3_counter:
            most_common_decimal3 = decimal3_counter.most_common(1)[0][0]
            most_common_count = decimal3_counter[most_common_decimal3]

            data_frame_list.append(most_common_decimal3)
            
            # 创建目标文件夹（使用最常见的后三位命名）
            target_folder = dest_path / most_common_decimal3
            target_folder.mkdir(parents=True, exist_ok=True)
            
            # 移动该分组内的所有文件
            for full_timestamp, file_path in items:
                dest_file = target_folder / file_path.name
                shutil.move(str(file_path), str(dest_file))
                # print(f"   ✓ {file_path.name} -> {most_common_decimal3}/")
                total_moved += 1
        else:
            print(f"   ⚠️ 无法提取后三位，跳过")
    
    print(f"\n✅ 完成！移动了 {total_moved} 个文件到 {dest_path}")

    # 清理移空后的相机名子文件夹
    removed_dirs = 0
    for sub_dir in source_path.iterdir():
        if sub_dir.is_dir() and not any(sub_dir.iterdir()):
            sub_dir.rmdir()
            removed_dirs += 1
    if removed_dirs:
        print(f"🗑️ 清理了 {removed_dirs} 个空的相机文件夹")

    return data_frame_list

def _extract_pose_from_json(file_path: Path) -> Dict:
    """
    从单个 JSON 文件提取位姿和时间戳信息
    
    Args:
        file_path: JSON 文件路径
    
    Returns:
        包含提取字段的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = _hugsim_expand(json.load(f))
    
    # 提取所需字段
    result = {
        # 外部时间戳
        'frame': data.get('frame'),
        
        # 额外的位姿时间戳
        'pose_timestamp': data['message']['timestamp'],
        # 位置信息
        'pos_x': data['message']['pos_smooth']['x'],
        'pos_y': data['message']['pos_smooth']['y'],
        'pos_z': data['message']['pos_smooth']['z'],
        
        # 姿态信息（欧拉角）
        'yaw': data['message']['yaw'],
        'pitch': data['message']['pitch'],
        'roll': data['message']['roll'],
}
    
    return result

def batch_extract_to_dataframe(json_dir: str, pattern: str = "*.json") -> pd.DataFrame:
    """
    批量提取 JSON 文件中的位姿数据到 DataFrame
    
    Args:
        json_dir: JSON 文件所在目录
        pattern: 文件匹配模式，默认 "*.json"
    
    Returns:
        包含所有数据的 DataFrame
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob(pattern))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    all_data = []
    
    for i, file_path in enumerate(json_files):
        try:
            data = _extract_pose_from_json(file_path)
            all_data.append(data)
                
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
    
    # 创建 DataFrame
    df = pd.DataFrame(all_data)
    
    # 按 frame 排序（如果存在）
    if 'frame' in df.columns:
        df = df.sort_values('frame').reset_index(drop=True)
    
    print(f"\n成功提取 {len(df)} 条记录")
    print(f"时间范围: {df['pose_timestamp'].min()} -> {df['pose_timestamp'].max()}")
    
    return df

def write_data_frame_json(images_dir: Path, df: pd.DataFrame, data_frame_list: list):
    '''
    写入每个文件夹data_frame.json(车辆位姿)
        进行了两种判断:
        1. 时间一致性判断,接受误差在半帧内
        2. 丢失位姿数据(全为0)跳过
    在clip文件夹写入data_frame_seq.json(需要按时间升序排列)
    '''
    
    images_dir = Path(images_dir)
    timestamp_list = data_frame_list
    
    data_frame_seq_items = []

    for idx, timestamp in enumerate(timestamp_list):
        try:


            idx = df['pose_timestamp'].searchsorted(timestamp)

            # 需要比较 idx 和 idx-1 哪个更接近
            if idx == 0:
                closest_idx = idx
            elif idx == len(df):
                closest_idx = idx - 1
            else:
                prev_diff = abs(df.loc[idx-1, 'pose_timestamp'] - float(timestamp))
                curr_diff = abs(df.loc[idx, 'pose_timestamp'] - float(timestamp))
                closest_idx = idx-1 if prev_diff < curr_diff else idx

            # 判断时间差是否在允许范围内,按0.05s计算(半帧)
            time_diff = abs(df.loc[closest_idx, 'pose_timestamp'] - float(timestamp))
            if time_diff > 0.05:
                continue

            data_row = df.iloc[closest_idx] # 只获取一次
            if (data_row.pos_x == 0 and data_row.pos_y == 0 and data_row.pos_z == 0 
                and data_row.yaw == 0 and data_row.pitch == 0 and data_row.roll == 0):

                print(f"跳过 {timestamp}: 位姿数据全为零")
                continue

            # /////////////////////data_frame.json/////////////////////
            # json写每个相机的vehicle_pose
            image_infos = []
            for cam_name in camera_names:

                data_row = df.iloc[closest_idx]

                if data_row.pos_x == 0 \
                    and data_row.pos_y == 0 \
                    and data_row.pos_z == 0 \
                    and data_row.yaw == 0 \
                    and data_row.pitch == 0 \
                    and data_row.roll == 0 :

                    continue

                image_info = {
                "camera_id": cam_name,
                "timestamp": data_row.pose_timestamp,
                "vehicle_pose": {
                    "x" : data_row.pos_x,
                    "y" : data_row.pos_y,
                    "z" : data_row.pos_z,
                    "yaw" : data_row.yaw,
                    "pitch" : data_row.pitch,
                    "roll" : data_row.roll
                    }
                }
                
                image_infos.append(image_info)

            data_frame_json = {
                "main_timestamp" : df.iloc[closest_idx]['pose_timestamp'],
                "image_infos": image_infos
            }
            # 写入 JSON 文件
            data_frame_json_file_name = 'data_frame.json'
            data_frame_json_file_path = images_dir / timestamp / data_frame_json_file_name 

            with open(data_frame_json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_frame_json, f, indent=4)

            # \\\\\\\\\\\\\\\\\\\\data_frame.json\\\\\\\\\\\\\\\\\\\\
            
            timestamp_dict = {
                "main_timestamp": df.iloc[closest_idx]['pose_timestamp'],
                "data_frame_path": timestamp,
            }
            
            # 检查 timestamp_dict 是否为空
            if not timestamp_dict or timestamp_dict.get('main_timestamp') is None:
                raise ValueError(f"timestamp_dict 为空或无效: {timestamp_dict}")
            
            data_frame_seq_items.append(timestamp_dict)

        except Exception as e:
            # 抛出包含详细信息的异常
            raise RuntimeError(f"处理第 {idx} 个 timestamp '{timestamp}' 时失败: {str(e)}") from e

    # 检查最终结果
    if not data_frame_seq_items:
        raise ValueError("data_frame_seq_items 为空，没有成功处理任何 timestamp")
    
    # 写入 data_frame_seq.json
    data_frame_seq_items.sort(key=lambda x: x['data_frame_path'])

    data_frame_seq_json = {"data_frame_seq_items": data_frame_seq_items}
    data_frame_seq_json_file_path = images_dir / 'data_frame_seq.json'
    
    try:
        with open(data_frame_seq_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_frame_seq_json, f, indent=4)
    except Exception as e:
        raise IOError(f"写入 data_frame_seq.json 失败: {str(e)}") from e
    
    print(f"===========成功将data_frame.json写入文件夹===========")
    print(f"成功处理 {len(data_frame_seq_items)} 个 timestamp")

def write_ego_trajectory(pose_proto_dir,output_dir):
    """将逐帧的 pose_proto JSON 文件汇总为一条自车轨迹文件。

    遍历 pose_proto_dir 目录下的所有 JSON 文件，从中提取平滑位置 (pos_smooth)
    和姿态角 (yaw/pitch/roll)，汇总为一个按文件名排序的列表，写入
    pose_proto_dir 上两级目录(解析后图片的同一个文件夹下)的 ego_trajectory.json。

    输出格式:
        [
            {
                "timestamp": <int, 时间戳>,
                "vehicle_pose": {
                    "x": <float>, "y": <float>, "z": <float>,
                    "yaw": <float>, "pitch": <float>, "roll": <float>
                }
            },
            ...
        ]

    Args:
        pose_proto_dir (Path): 存放逐帧 pose_proto JSON 文件的目录。

    Returns:
        None: 结果写入 {pose_proto_dir}/../../ego_trajectory.json。

    Note:
        JSON 解析失败或缺少预期字段的文件会被跳过并打印错误信息，
    """
    # 列出所有读取pose_proto_dir目录下所有的json
    json_files = sorted(os.listdir(pose_proto_dir))

    data_list = []
    # 依次读取json
    for file in json_files:
        try:
            with open(pose_proto_dir / file, 'r', encoding='utf-8') as f:
                data = _hugsim_expand(json.load(f))
            
            msg = data['message']
            pose = msg['pos_smooth']

            data_list.append({'timestamp': msg['timestamp'],
                              'vehicle_pose': {
                            'x': pose['x'],
                            'y': pose['y'],
                            'z': pose['z'],
                            'yaw': msg['yaw'],
                            'pitch': msg['pitch'],
                            'roll': msg['roll']}
            })
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {file} -> {e}")
        except Exception as e:
            print(f"读取文件出错: {file} -> {e}")

    ego_trajectory_output_path = output_dir / "ego_trajectory.json"
    with open(ego_trajectory_output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"ego trajectory.json已保存到: { ego_trajectory_output_path}\n 存在: {ego_trajectory_output_path.exists()}")  
# ====================图片组织模块=======================

# ===================数据完整性验证模块==================
# 功能1 : 验证文件夹中文件完整
def required_files_validation(
        IMG_OUTPUT_DIR: Path, 
        data_frame_list: list,
        required_files_list: list) -> list:
    
    missing_list = []

    for folder_name in data_frame_list:
        folder_path = IMG_OUTPUT_DIR / folder_name

        folder_file_num = len(os.listdir(folder_path))
        if folder_file_num != len(required_files_list):
            print(f"文件夹{folder_name}中文件数量:{folder_file_num}, 应当有数量:{len(required_files_list)}")
            missing_list.append((folder_name, folder_path))

    return missing_list

# 功能2 : 数据不完整的文件夹删除,data_frame_seq.json相关内容删除
def delete_incomplete_folders(
        IMG_OUTPUT_DIR: Path,
        missing_list: List[Tuple[str, Path]]) -> None:
    """
    删除数据不完整的文件夹，并更新 data_frame_seq.json
    """
    # 读取 data_frame_seq.json
    data_frame_seq_path = IMG_OUTPUT_DIR / "data_frame_seq.json"
    with open(data_frame_seq_path, "r") as f:
        data_frame_seq = _hugsim_expand(json.load(f))
    
    # 获取需要删除的文件夹名称列表
    folders_to_delete = [folder_name for folder_name, _ in missing_list]
    
    # 过滤掉需要删除的项
    new_items = [
        item for item in data_frame_seq['data_frame_seq_items']
        if item['data_frame_path'] not in folders_to_delete
    ]
    
    # 更新 JSON 数据
    data_frame_seq['data_frame_seq_items'] = new_items
    
    # 删除文件夹
    deleted_folders = []
    for folder_name, folder_path in missing_list:
        try:
            if folder_path.exists():
                shutil.rmtree(folder_path)  # 使用 rmtree 删除目录
                deleted_folders.append(folder_name)
                print(f"🗑️ 已删除文件夹: {folder_name}")
            else:
                print(f"⚠️ 文件夹已不存在: {folder_name}")
        except Exception as e:
            print(f"❌ 删除失败 {folder_name}: {e}")
    
    # 写入更新后的 JSON 文件
    with open(data_frame_seq_path, "w") as f:
        json.dump(data_frame_seq, f, indent=4)
    
    return None

# ===================数据完整性验证模块==================


class LidarToFrameAligner:
    """
    根据绝对时间戳将点云文件对齐到图像帧文件夹
    图像文件夹名与点云文件名均为同一时基的绝对时间戳(秒, 图像为 middle_row_exposure)
    """

    def __init__(
        self,
        data_frame_list: list,
        pcd_source_dir: str,
        frames_root_dir: str,
    ):
        """
        Args:
            data_frame_list: 图像帧文件夹名列表(绝对时间戳, 秒)
            pcd_source_dir: 存放原始点云文件的目录
            frames_root_dir: 图像帧文件夹的根目录（包含各 timestamp 子文件夹）
        """
        self.ts_folder_list = np.sort(data_frame_list)
        self.ts_folder_float = np.array(self.ts_folder_list, dtype=float)
        self.pcd_source_dir = Path(pcd_source_dir)
        self.frames_root_dir = Path(frames_root_dir)

        #验证数据完整打印信息
        self._validate_img_num()
        self._validate_pcd_num()

        # 加载并排序图像帧信息
        self.df_img_folder = self._build_img_folder_df()

        # 加载并排序点云信息
        self.pcd_df = self._load_pcd_timestamps()

        # 点云与图像文件夹对齐
        self.aligned_df = self._pcd_img_alignment()


    def _validate_pcd_num(self):

        pcd_files = [f for f in os.listdir(self.pcd_source_dir) if f.endswith('.pcd')]

        pcd_ts_list = []

        for pcd_file in pcd_files:
            # 从文件名提取pcd的绝对时间戳(秒), 与图像 middle_row_exposure 同时基
            # 格式: pointcloud-LDR_FRONT-1751434405.5515.pcd
            match = re.search(r'-(\d+\.\d+)\.pcd$', pcd_file)
            if not match:
                print(f"警告: 无法解析文件名 {pcd_file}，跳过")
                continue

            pcd_ts_list.append(float(match.group(1)))

        pcd_ts_array = np.sort(np.array(pcd_ts_list))
        time_range = pcd_ts_array[-1] - pcd_ts_array[0]

        print("=============点云数据统计===============")
        print(f"点云时间范围:\n     起: {pcd_ts_array[0]:.4f} 秒\n     止:{pcd_ts_array[-1]:.4f} 秒")
        print(f"共{time_range:.1f}秒\n应有点云 : {round(time_range / 0.1) + 1}帧")

        print(f"实有点云 : {len(pcd_files)}帧")

        if len(pcd_files) == round(time_range / 0.1) + 1:
            print("=============点云数据未缺失=============")
        else:
            print("=============点云数据有缺失=============")

        return None

    def _validate_img_num(self):

        # 图像帧文件夹名即绝对时间戳(秒)
        img_timestamp_array = self.ts_folder_float

        time_range_s = img_timestamp_array[-1] - img_timestamp_array[0]

        # 根据时间范围估算期望帧数 (10Hz)
        expected_fps = 10
        expected_frame_count = int(time_range_s * expected_fps) + 1
        actual_frame_count = len(self.ts_folder_list)

        # 打印结果
        print("\n=============图片数据统计===============")
        print(f"图片 timestamp 范围: \n     起: {img_timestamp_array[0]:.4f} 秒\n     止: {img_timestamp_array[-1]:.4f} 秒")
        print(f"时间跨度: {time_range_s} 秒\t约{time_range_s:.1f} 秒")

        print(f"应有图片: {expected_frame_count} 帧")
        print(f"实有图片: {actual_frame_count} 帧")

        if expected_frame_count == actual_frame_count:
            print("============图片数据无缺失=============")

        else:
            print("============图片数据有缺失=============\n")

        return None

    def _find_abnormal_gaps(self):
        '''
        仅用于检测归类后的图片是否有缺帧情况
        '''
        ts = np.sort(np.array(self.ts_folder_list, dtype=float))
        diffs = np.diff(ts)

        expected_interval = 0.1
        tolerance_ratio = 0.1
        tol = expected_interval * tolerance_ratio

        abnormal = []
        for i, d in enumerate(diffs):
            if abs(d - expected_interval) > tol:
                missing = int(round(d / expected_interval)) - 1
                abnormal.append((i, ts[i], ts[i+1], d, missing))

        # 打印结果
        if not abnormal:
            print("✅ 未检测到异常间隔，所有时间间隔正常")
        else:
            print(f"⚠️ 检测到 {len(abnormal)} 处异常间隔：\n")
            for idx, (i, t_start, t_end, gap, missing) in enumerate(abnormal, 1):
                print(f"异常 {idx}:")
                print(f"  位置索引: {i}")
                print(f"  时间区间: {t_start:.3f}s |----| {t_end:.3f}s")
                print(f"  实际间隔: {gap:.3f}s ({gap*1000:.2f}ms)")
                print(f"  预期间隔: {expected_interval}s (100ms)")
                print(f"  偏差: {(gap - expected_interval)*1000:.2f}ms")
                print(f"  缺失帧数: {missing}")
                if missing > 0:
                    # 列出缺失的理论时间戳
                    missing_ts = [t_start + (j+1)*expected_interval for j in range(missing)]
                    print(f"  缺失时间戳: {', '.join(f'{t:.3f}s' for t in missing_ts)}")
                print()

        return abnormal

    def _build_img_folder_df(self) -> pd.DataFrame:
        """基于图像帧文件夹名(绝对时间戳)构建 DataFrame, 并插入缺帧占位行"""
        df_img_folder = pd.DataFrame({
            'folder_name': self.ts_folder_list ,
            'folder_path': [self.frames_root_dir / folder_name for folder_name in self.ts_folder_list]
        })

        abnormal = self._find_abnormal_gaps()
        for (i, _, _, _, missing) in abnormal:
            # 在 df_img_folder 的第 i+1 到 i+missing 行插入 pd.NA
            insert_pos = i + 1  # 插入位置
            # 创建空行 DataFrame
            empty_rows = pd.DataFrame([[pd.NA] * len(df_img_folder.columns)] * missing,
                                    columns=df_img_folder.columns)
            # 分割并重新拼接
            df_img_folder = pd.concat([
                df_img_folder.iloc[:insert_pos],
                empty_rows,
                df_img_folder.iloc[insert_pos:]
            ], ignore_index=True)

        return df_img_folder

    def _load_pcd_timestamps(self) -> pd.DataFrame:
        """扫描点云目录，解析每个文件名的绝对时间戳(秒, 与图像 middle_row_exposure 同时基)，返回排序后的 DataFrame"""
        pcd_files = list(self.pcd_source_dir.glob("*.pcd"))
        if not pcd_files:
            raise FileNotFoundError(f"在 {self.pcd_source_dir} 中未找到任何 .pcd 文件")

        records = []
        for pcd_path in pcd_files:
            # 从文件名中提取绝对时间戳(秒)
            match = re.search(r"-(\d+\.\d+)\.pcd$", pcd_path.name)

            if not match:
                print(f"⚠️ 跳过无法解析的文件: {pcd_path.name}")
                continue

            records.append({
                "pcd_ts": float(match.group(1)),
                "pcd_src_path": pcd_path,
                "filename": pcd_path.name,
            })

        if not records:
            raise ValueError("没有成功解析出任何点云的时间戳，请检查文件名格式")

        df = pd.DataFrame(records)
        df.sort_values("pcd_ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _pcd_img_alignment(self):
        """图像文件夹名与点云文件名均为同一时基的绝对时间戳(秒),
        每帧点云按时间最近邻归入图像帧文件夹, 时间差超过半帧(0.05s)视为无归属
        """
        pcd_df = self.pcd_df
        folder_ts = self.ts_folder_float
        folder_names = self.ts_folder_list
        folder_paths = [self.frames_root_dir / name for name in folder_names]

        records = []
        for _, pcd_row in pcd_df.iterrows():
            pcd_ts = pcd_row['pcd_ts']
            idx = np.searchsorted(folder_ts, pcd_ts)

            # 比较 idx 和 idx-1 哪个更接近
            best_idx = None
            best_diff = None
            for cand in (idx - 1, idx):
                if 0 <= cand < len(folder_ts):
                    diff = abs(folder_ts[cand] - pcd_ts)
                    if best_diff is None or diff < best_diff:
                        best_diff, best_idx = diff, cand

            if best_idx is None or best_diff > 0.05:
                # 无归属点云: folder 置空, 由 move_pcd_files(delete_invalid=True) 删除
                records.append({
                    "folder_name": pd.NA,
                    "folder_path": pd.NA,
                    "pcd_ts": pcd_ts,
                    "pcd_src_path": pcd_row['pcd_src_path'],
                    "filename": pcd_row['filename'],
                })
                print(f"⚠️ 点云 {pcd_row['filename']} (ts={pcd_ts:.4f}) 与最近图像帧相差 {best_diff:.4f}s, 视为无归属")
            else:
                records.append({
                    "folder_name": folder_names[best_idx],
                    "folder_path": folder_paths[best_idx],
                    "pcd_ts": pcd_ts,
                    "pcd_src_path": pcd_row['pcd_src_path'],
                    "filename": pcd_row['filename'],
                })

        pcd_assigned = pd.DataFrame(records)
        # 以图像帧文件夹为主轴(含缺帧占位行), 关联已分配的点云
        result = self.df_img_folder.merge(pcd_assigned, on=['folder_name', 'folder_path'], how='left')

        return result


    def move_pcd_files(self, delete_invalid=False):
        result = self.aligned_df
        moved_count = 0

        for idx, row in result.iterrows():

            folder_path = row.get('folder_path')
            pcd_path = row.get('pcd_src_path')

            if pd.notna(folder_path) and pd.notna(pcd_path):
                try:   
                    shutil.move(str(pcd_path), str(folder_path))
                    moved_count += 1
                    # print(f"✓ 移动: {Path(pcd_path).name} -> {folder_path}")
                except Exception as e:
                    print(f"✗ 移动失败: {e}")
            # 因为后面还会检查文件夹中数据完整性,所以这里不需要提前删除没有点云的文件夹了
            # elif pd.notna(folder_path) and pd.isna(pcd_path):
            #     if delete_invalid:
            #         try:
            #             folder = Path(folder_path)
            #             if folder.exists():
            #                 shutil.rmtree(folder) # 强制删除文件夹
            #                 print(f"✓ 删除文件夹: {folder_path}")
            #         except Exception as e:
            #             print(f"✗ 删除失败: {e}")
            elif pd.notna(pcd_path) and pd.isna(folder_path):
                if delete_invalid: 
                    try:
                        Path(pcd_path).unlink()
                        print(f"✓ 删除无归属点云: {pcd_path}")
                    except Exception as e:
                        print(f"✗ 删除失败: {e}")

        print(f"\n总结: 成功移动 {moved_count} 个点云")
        return moved_count

        
def main():

    parser = argparse.ArgumentParser(description='图片,点云,dataframe.json组织')
    parser.add_argument('--parsed_data_root', default="home/data/parsed/", help='输入存放各场景数据的目录')
    parser.add_argument('--scene_id', type=str, help='数据包名')
    parser.add_argument('--time_range', '-t', nargs=2,  type=int, metavar=('START', 'END'), 
                        required=False, help='切片的时间范围, 如 -t 30 60')
    
    args = parser.parse_args()

    slice_start_time ,slice_end_time = args.time_range

    scene_id_s_e = f"{args.scene_id}_{slice_start_time}_{slice_end_time}"

    # ========== 用户配置区 ==========

    PARSED_SCENE_DIR = Path(args.parsed_data_root) / args.scene_id
    
    PCD_OUTPUT_DIR = PARSED_SCENE_DIR / f"pcd_{slice_start_time}_{slice_end_time}"
    IMG_OUTPUT_DIR = PARSED_SCENE_DIR / scene_id_s_e  
 
    VEH_POS_JSON_DIR = PARSED_SCENE_DIR / f"lite_msg_{slice_start_time}_{slice_end_time}" / "pose_proto"

    # =============本地调试使用===================

    # IMG_OUTPUT_DIR = Path("/home/chery/Projects/Data/20260526_145008_QCJPSD851972/20260526_145008_QCJPSD851972_3096_3121/")
    # PCD_OUTPUT_DIR = Path("/home/chery/Projects/Data/20260526_145008_QCJPSD851972/pcd_3096_3121/")
    # VEH_POS_JSON_DIR = Path("/home/chery/Projects/Data/20260526_145008_QCJPSD851972/lite_msg_3096_3121/pose_proto")

    # ==========================================

    # 将解析图片按时间戳归入文件夹
    data_frame_list = organize_images_by_majority_decimal3(IMG_OUTPUT_DIR)   # data_frame_list就是图片文件夹名的列表

    # 提取数据到dataframe
    veh_pos_df = batch_extract_to_dataframe(VEH_POS_JSON_DIR)

    # 写data_frame.json并按时间戳归入文件夹
    write_data_frame_json(IMG_OUTPUT_DIR, veh_pos_df, data_frame_list)

    # 整合ego_pose
    write_ego_trajectory(VEH_POS_JSON_DIR, output_dir=IMG_OUTPUT_DIR)

    aligner = LidarToFrameAligner(
        data_frame_list=data_frame_list,
        pcd_source_dir=PCD_OUTPUT_DIR,
        frames_root_dir=IMG_OUTPUT_DIR,
    )

    aligner.move_pcd_files(delete_invalid=True)

    missing_list = required_files_validation(
    IMG_OUTPUT_DIR, 
    data_frame_list, 
    required_files_list)

    if missing_list:
        delete_incomplete_folders(IMG_OUTPUT_DIR, missing_list)


if __name__ == "__main__":
    main()



'''
DONE    一. 验证点云数据与自身时间的完整性: 统计relative首尾时间戳,计算应当有多少帧点云,查看实际有多少帧点云
DONE    二. 验证图像数据与自身时间的完整性: 统计文件夹首尾timestamp,计算应当有多少帧图片,查看实际有多少图片文件夹(可能有中中间缺失)
DONE    三. 根据relative_ts对应点云和图片的第一帧,然后逐步将每一个点云放到相应的folder中
            (1) 可以处理第一帧无法天然对齐的两种情况
            (2) 可以处理图片数据帧不连续的异常处理


    1. 用点云去对第一帧图片,就按前五帧算,注意到这个数列必然是增数列
        1.1 img_ts[0:6] - pcd_ts[0]的结果可能是
            i. [-0.2, -0.1, 0.0, 0.1, 0.2], 这种情况下 第一帧点云在第一帧图片之后
                    pcd =       [1, 2, 3, 4, 5]
                    img = [a, b, c, d, e]

            ii. [0.0, 0.1, 0.2, 0.3, 0.4] 这种情况下 第一帧点云对应第一帧图片
                    pcd = [1, 2, 3, 4, 5]
                    img = [a, b, c, d, e]

            iii. [0.1, 0.2, 0.3, 0.4 0.5] 这种情况下 第一帧点云在第一帧图片之前
                因此反过来,用图片去对点云,情况如下:
                    pcd = [1, 2, 3, 4, 5]
                    img =    [a, b, c, d, e]
                    得到上面i或者ii的结果
    2. 得到第x帧点云对应了第x帧图片,因此第x帧点云应移动到第x个文件夹
            先要对data_frame_list缺失帧的位置占上
                    pcd =       [1, 2, 3, 4, 5]
                    img = [a, b, c, d, 0, e]
                 folder = [A, B, C, D, 0, E]
                  
                  flag0 = [1, 1, 1, 1, 0, 1]          初始化flag,表示那些folder是可以放入点云的          
                  flag1 = [0, 0, 1, 1, 0, 1]          判别后,哪些folder是不需要放入点云的
    3. 现在已知一个data_frame_list里面时间戳,这些时间戳是存放图片与点云的文件夹的名字
        图像文件夹名与点云文件名均为同一时基的绝对时间戳(图像为 middle_row_exposure),
        每帧点云按时间最近邻归入图像帧文件夹
'''