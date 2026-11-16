# StfTools — QCraft STF 数据预处理工具

本目录包含 QCraft 场景数据从原始 STF/QSTF 包到训练可用格式的预处理脚本。
以下脚本按实际 pipeline 执行顺序编排说明。

> 注意：当前脚本通过 `proto_from_descriptors.py` + `proto_descriptors.bin` 动态加载 protobuf 消息，**不再依赖 `onboard/`、`offboard/`、`common/` 等 proto 源码目录**。

---

## 环境依赖

### Python 依赖

```bash
pip install protobuf>=3.19.4 cramjam>=2.0.0 zstandard>=0.19.0
```

### 可选依赖

| 脚本 | 可选包 | 说明 |
|---|---|---|
| `stf_extract_h265_to_jpg.py` | `av` | H.265/H.264 软解码 |
| `stf_extract_h265_to_jpg.py` | `Pillow` | 图像质量/灰帧检测 |
| `img_pcd_organize.py` | `pandas` | 时间戳对齐 |
| `3dbox_visualization.py` | `cv2`, `moviepy`, `matplotlib` | 可视化 |

### 系统依赖

- **FFmpeg**：用于 H.265/H.264 硬件/软件解码辅助。
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

---

## 脚本执行流程

完整数据准备顺序：

```text
1. download_snippets.py
2. snippets_merging.py
3. stf_extract_h265_to_jpg.py
4. stf_extract_pointcloud.py
5. stf_to_json_full.py
6. extract_sensor_params.py
7. img_pcd_organize.py
```

---

## 1. 从 OBS snippets 下载数据

**脚本**：`download_snippets.py`

当完整大包 `camera_jpg_img.stf` / `lidar_data.stf` / `lite_msg.stf` 不存在时，从 OBS snippets 中查找并下载覆盖目标时间段的片段。

```bash
python3 download_snippets.py \
    --scene_id 20260526_145008_QCJPSD851972 \
    --clip_start_time 350 \
    --clip_end_time 380 \
    --output_dir /path/to/raw/data \
    --obsutil tools/obsutil_linux_amd64_5.8.3/obsutil \
    --endpoint obs.cn-east-4.myhuaweicloud.com
```

**参数说明**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--scene_id` | 是 | 场景 ID |
| `--clip_start_time` | 是 | 切片起始时间（秒） |
| `--clip_end_time` | 是 | 切片结束时间（秒） |
| `--output_dir` | 是 | 下载输出目录 |
| `--obsutil` | 是 | obsutil 可执行文件路径 |
| `--endpoint` | 是 | OBS endpoint |

**输出**：`output_dir/<scene_id>/<scene_id>_<start>_<end>/download/`

---

## 2. 合并 snippets 并按时间切片

**脚本**：`snippets_merging.py`

当使用 snippets 下载时，多个 snippet 之间可能存在时间重叠，需要先合并并截取目标时间段。

```bash
python3 snippets_merging.py \
    "--download_dir" /path/to/raw/data/<scene_id>/<scene_id>_<start>_<end>/download \
    "--clip_start_time" 350 \
    "--clip_end_time" 380 \
    "--output" /path/to/raw/data/<scene_id>/<scene_id>_<start>_<end>
```

**参数说明**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--download_dir` | 是 | snippets 下载目录 |
| `--clip_start_time` | 是 | 切片起始时间（秒） |
| `--clip_end_time` | 是 | 切片结束时间（秒） |
| `--output` | 是 | 合并后输出目录 |

**输出**：输出目录下生成 `camera_jpg_img_<start>_<end>.stf`、`lidar_data_<start>_<end>.stf`、`lite_msg_<start>_<end>.stf`。

---

## 3. 提取图像（H.265/JPEG 自动识别）

**脚本**：`stf_extract_h265_to_jpg.py`

从 `camera_jpg_img_*.stf` 中提取图像，保存为 JPEG。支持 JPEG、H.265、H.264 自动识别。

```bash
python3 stf_extract_h265_to_jpg.py \
    /path/to/snippets/camera_jpg_img_350_380.stf \
    --output /path/to/parsed/<scene_id>/<scene_id>_350_380
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `input` | 是 | - | 输入 STF 文件路径 |
| `--output` / `-o` | 否 | `images` | 输出目录 |
| `--threads` / `-t` | 否 | 4 | 写入线程数 |
| `--decode-threads` / `-d` | 否 | 4 | 解码线程数（H.265/H.264） |
| `--quality` / `-q` | 否 | 95 | JPEG 质量（1-100） |
| `--limit` / `-l` | 否 | - | 最多提取帧数 |
| `--skip-gray` | 否 | False | 跳过灰噪/异常帧 |

**输出**：`output/<camera_name>/<timestamp>.jpg`

> 注意：H.265/H.264 解码需要安装 `av`（PyAV）和 FFmpeg。灰帧检测需要 `Pillow`。

---

## 4. 提取点云

**脚本**：`stf_extract_pointcloud.py`

从 `lidar_data_*.stf` 中提取激光雷达点云，自动读取 `lite_run_info.pb.bin` 中的内外参，并做运动补偿。

```bash
python3 stf_extract_pointcloud.py \
    /path/to/snippets \
    lidar_data_350_380.stf \
    --output /path/to/parsed/<scene_id>/pcd_350_380 \
    --apply-extrinsics
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `run_dir` | 是 | - | 包含 STF 和 `lite_run_info.pb.bin` 的目录 |
| `stf_file` | 是 | - | lidar STF 文件名 |
| `--output` / `-o` | 否 | `pointclouds` | 输出目录 |
| `--apply-extrinsics` | 否 | False | 将点云应用外参变换到车辆坐标系 |
| `--binary` / `-b` | 否 | False | 输出 binary 格式 PCD |
| `--limit` / `-l` | 否 | - | 最多提取帧数 |
| `--verbose` / `-v` | 否 | False | 详细输出 |

**输出**：`output/pointcloud_*.pcd`

---

## 5. 解析 LiteMsg 到 JSON

**脚本**：`stf_to_json_full.py`

将 `lite_msg_*.stf` 中的 protobuf 消息完整解析为 JSON，包含 pose、chassis、objects、localization 等话题。

```bash
python3 stf_to_json_full.py \
    /path/to/snippets/lite_msg_350_380.stf \
    --output /path/to/parsed/<scene_id>
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `input` | 是 | - | 输入 STF 文件或 run 目录 |
| `--output` / `-o` | 否 | `output` | 输出目录 |
| `--threads` / `-t` | 否 | 4 | 线程数 |
| `--workers` / `-w` | 否 | 4 | 进程池 worker 数 |
| `--topics` | 否 | - | 只导出指定话题，逗号分隔 |

**输出**：`output/lite_msg_*/frame_*.json`

---

## 6. 提取传感器参数

**脚本**：`extract_sensor_params.py`

从 `lite_run_info.pb.bin` 中提取相机内外参、LiDAR 外参、车辆参数，保存为 JSON。

```bash
python3 extract_sensor_params.py \
    /path/to/snippets \
    --output /path/to/parsed/<scene_id>/<scene_id>_350_380
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `run_dir` | 是 | - | 包含 `lite_run_info.pb.bin` 的目录 |
| `--output` / `-o` | 否 | 当前目录 | 输出目录 |

**输出**：输出目录下的 `camera_params.json`、`lidar_params.json`、`vehicle_params.json` 等。

---

## 7. 组织图像与点云文件

**脚本**：`img_pcd_organize.py`

按时间戳对齐图像和点云，生成训练框架需要的 `data_frame.json` 和目录结构。

```bash
python3 img_pcd_organize.py \
    --parsed_data_root /path/to/parsed \
    --scene_id 20260526_145008_QCJPSD851972 \
    --time_range 350 380
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--parsed_data_root` | 是 | - | 已解析数据根目录 |
| `--scene_id` | 是 | - | 场景 ID |
| `--time_range` | 是 | - | 起始和结束时间（秒） |

**输出**：在 `parsed_data_root/<scene_id>/<scene_id>_<start>_<end>/` 下生成对齐后的文件结构。

---
## 8. 3D 标注框可视化

`third_party/StfTools/3dbox_visualization.py` 用于将 `label_pred/` 中的 3D 标注框反投影到相机图像上，并生成：

- 各相机带 3D 框的逐帧图像
- 每个相机的视频
- 6 路相机网格视频
- BEV 2D 框视频
- 可选的 BEV + 相机网格拼接视频

### 用法

从按时间戳组织的图像目录读取（默认模式）：

```bash
python3 third_party/StfTools/3dbox_visualization.py \
    --data_dir /home/chery/Projects/Data/parsed/20260525_155152_QCJPSD851972/20260525_155152_QCJPSD851972_2570_2590/ \
    --output_folder visualization
```

从扁平化的重建结果图片目录读取（图片命名为 `000000_006.png` 形式，对应 `frame_idx` 和相机编号）：

```bash
python3 third_party/StfTools/3dbox_visualization.py \
    --data_dir /path/to/parsed_scene \
    --output_folder visualization \
    --img_input_dir /path/to/full_set_40000_gt_rgbs
```

### 坐标系说明

脚本内部采用如下约定：

- **车辆坐标系**：X-前，Y-左，Z-上
- **相机坐标系**：X-右，Y-下，Z-前（OpenCV/ROS 标准）
- 3D 框标注 `psr` 中的 `position`、`rotation`、`scale` 均位于车辆坐标系下

### 依赖

```bash
pip install opencv-python moviepy matplotlib
```
---

## 组织好的数据案例

下面是一个已解析并组织好的 QCraft 场景示例，路径为：

`/home/chery/Projects/Data/parsed/20260525_155152_QCJPSD851972/20260525_155152_QCJPSD851972_2570_2590/`

该目录下包含：

```shell
20260525_155152_QCJPSD851972_2570_2590/
├── 1779698088.819/              # 每一帧的时间戳文件夹
│   ├── CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110-1779698088.8194.jpg
│   ├── CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60-1779698088.8194.jpg
│   ├── CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30-1779698088.8194.jpg
│   ├── CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15-1779698088.8194.jpg
│   ├── CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99-1779698088.8197.jpg
│   ├── CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99-1779698088.8197.jpg
│   ├── CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30-1779698088.8197.jpg
│   ├── CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99-1779698088.8197.jpg
│   ├── CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99-1779698088.8197.jpg
│   ├── CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30-1779698088.8197.jpg
│   ├── CAM_PBQ_REAR_RESET_OPTICAL_H50-1779698088.8197.jpg
│   ├── pointcloud-LDR_FRONT-1779698088.8697.pcd
│   └── data_frame.json            # 当前帧的图像时间戳信息
├── 1779698088.919/
├── ...
├── camera_params.json             # 相机内外参（车辆坐标系）
├── data_frame_car_info.json       # 车辆/自车信息
├── data_frame_seq.json            # 时间戳序列与文件夹映射
├── vehicle_pose.json              # 每一帧的自车位姿
├── ego_masks/                     # 自车 ego mask
├── label_pred/                    # 自动/人工提取的 3D box 标注
│   ├── 1779698088.819.json
│   ├── 1779698088.919.json
│   └── ...
├── label_pred0/                   # 备用/其他来源的 3D box 标注
├── label_pred1/
└── visualization/                 # 3dbox_visualization.py 输出
    ├── CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/
    ├── CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60/
    ├── ...
    ├── bev_frames/
    ├── bev_video.mp4
    ├── grid_video.mp4
    ├── final_video.mp4
    └── CAM_*.mp4

```

主要文件说明：

| 文件/目录 | 说明 |
|---|---|
| `camera_params.json` | 各相机外参（`camera_to_vehicle_extrinsics`）与内参（`intrinsics`） |
| `data_frame_seq.json` | `data_frame_seq_items` 列表，每项包含 `main_timestamp` 与对应的时间戳文件夹名 |
| `vehicle_pose.json` | 每帧自车在世界坐标系下的位姿 `x, y, z, yaw, pitch, roll` |
| `label_pred/` | 每帧一个 `.json`，包含该帧所有 3D 标注框 `psr`（position/rotation/scale） |
| `<timestamp>/data_frame.json` | 单帧内各相机图像的实际时间戳列表 |



## 公共内部模块

以下文件不直接作为入口脚本运行，但为上述脚本提供公共能力：

| 文件 | 作用 |
|---|---|
| `stf_reader.py` | 内存式 STF/QSTF 读取器，支持 Snappy/Zstandard 解压 |
| `stf_reader_stream.py` | 流式 STF 读取器，低内存处理大文件 |
| `stf_split_stream.py` | STF 流式切分与写入工具 |
| `proto_from_descriptors.py` | 从 `proto_descriptors.bin` 动态加载 protobuf 消息类 |
| `proto_descriptors.bin` | 打包后的 protobuf FileDescriptorSet |

---

## 注意事项

1. **工作目录**：所有脚本默认在仓库根目录执行。`scripts/qcraft/data_preparation.sh` 已经按正确顺序编排。
2. **PYTHONPATH**：脚本内部会通过 `sys.path.insert(0, ...)` 将自身目录加入路径，因此可以直接从仓库根目录调用。
3. **protobuf 依赖**：
   - 不要删除 `proto_descriptors.bin`。
   - 不需要 `onboard/`、`offboard/`、`common/` 等目录即可运行当前 pipeline。
4. **H.265 解码**：
   - 如果未安装 `av`，H.265/H.264 片段会跳过或报错。
   - FFmpeg 是系统级依赖，需单独安装。
5. **OBS 下载**：
   - `download_snippets.py` 依赖华为云 OBS 认证环境变量（如 `AK`、`SK`），不要硬编码在脚本里。
6. **内存管理**：
   - 大文件处理优先使用 `*_stream.py` 系列脚本，避免 OOM。

---

## 快速验证

执行以下最小流程验证环境：

```bash
# 进入仓库根目录
export PYTHONPATH=$(pwd)

# 验证 protobuf 动态加载
python3 - <<'PY'
import sys
sys.path.insert(0, 'third_party/StfTools')
from proto_from_descriptors import get_message_class
print(get_message_class('qcraft.LiteMsgWrapper'))
PY

# 验证 STF 读取器
python3 - <<'PY'
import sys
sys.path.insert(0, 'third_party/StfTools')
from stf_reader import StfReader
print(StfReader)
PY
```

如果以上命令无报错，说明核心依赖已就绪。
