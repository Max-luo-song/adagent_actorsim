#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
################################################################################
# 公共变量和函数
################################################################################


# 公共变量
################################################################################

# OBS过滤
OBS_FILTER='\[[=>_ \-]+\]|[0-9]+\.[0-9]+%|tps:|Succeed count|Failed count|Succeed bytes|Metrics|Task id| [0-9]+/[0-9]+ |[0-9]+\.[0-9]+KB/s|[0-9]+\.[0-9]+MB/[0-9]+\.[0-9]+MB'

# OBS配置
ENDPOINT="obs.cn-east-4.myhuaweicloud.com"
OBSUTIL="tools/obsutil_linux_amd64_5.8.3/obsutil"

# 数据根目录
raw_data_root="/nas/scenario_output/zhangyingjun/data/raw"
parsed_data_root="/nas/scenario_output/zhangyingjun/data/parsed"
processed_data_root="/nas/scenario_output/zhangyingjun/data/processed"

# 训练配置
start_timestep=0
end_timestep=-1
gpu=0
segformer_path=/opt/mmsegmentation
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
lidar_type="lidar" # lidar（运动补偿前）/visual（纯视觉）
output_root="/nas/scenario_output/zhangyingjun/data/output"
config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/11cams_${lidar_type}"
extra_config_info="" # 额外信息，如 depth_loss

# 相机分组训练配置
group_execution_mode="serial"  # serial / parallel

train_group0_views=2,3,5,9,12
train_group1_views=0,1,6,7,10,11
eval_group0_views=2,3,5,9,12
eval_group1_views=0,1,6,7,10,11
num_view_groups=2

group0_gpu=0
group1_gpu=0

eval_downscale_when_loading="[1]"
view_order=0,1,2,3,5,6,7,9,10,11,12
fps=10

# LiDAR训练配置
train_lidar_iterations=3000
max_depth=100
block_size=50

# 公共函数
################################################################################

# 检查文件是否存在
check_file_exists() {
    local file_path="$1"
    local file_desc="${2:-文件}"
    
    if [ ! -f "$file_path" ]; then
        echo "=========================================="
        echo "✗ 错误: 未找到${file_desc}"
        echo "  Pipeline 已终止"
        echo "=========================================="
        return 1
    fi
    echo "=========================================="
    echo "✓ 步骤成功: 相机训练"
    echo "=========================================="
    return 0
}

# 检查目录是否为空
check_dir_not_empty() {
    local dir_path="$1"
    local dir_desc="${2:-目录}"
    
    if [ -z "$(ls -A "$dir_path" 2>/dev/null)" ]; then
        echo "=========================================="
        echo "✗ 错误: ${dir_desc}为空"
        echo "  Pipeline 已终止"
        echo "=========================================="
        return 1
    fi
    
    echo "=========================================="
    echo "✓ 步骤成功: LiDAR训练"
    echo "=========================================="
    return 0
}

# 从 OBS 自动获取最新的手工 3D 标注（label_pred）
download_manual_labels_if_exists() {
    local clip_name="$1"
    local parsed_root="$2"
    local scene_id="$3"
    local version="$4"
    
    # OBS 基础路径
    local obs_scene_root="obs://hwcn-hd2-ad-simulation/scene_reconstruction/${clip_name}/label/"
    local local_label_dir="$(hugsim_source_path "${parsed_root}/${version}/${scene_id}/${clip_name}/label_pred")"
    
    echo "[INFO] Checking manual labels at ${obs_scene_root}"
    
    # 获取所有日期子目录（6位数字）排序
    local ls_output
    ls_output=$("${OBSUTIL}" ls "${obs_scene_root}" -e="${ENDPOINT}" 2>/dev/null)
    local date_dirs
    date_dirs=$(echo "$ls_output" | grep -oE "[0-9]{6}/" | sed 's#/##' | sort -u -n)
    
    if [ -z "$date_dirs" ]; then
        echo "[INFO] No date directories found, will run 3dbox_extract"
        return 1
    fi
    
    # 取最新（即最大数字）
    local latest_date
    latest_date=$(echo "$date_dirs" | tail -1)
    echo "[INFO] Found date directories: $(echo $date_dirs | tr '\n' ' '), latest is ${latest_date}"
    
    local obs_label_path="${obs_scene_root}${latest_date}/label_pred"
    
    # 检查 label_pred 目录下是否有文件
    local check_output
    check_output=$("${OBSUTIL}" ls "${obs_label_path}/" -s -e="${ENDPOINT}" 2>/dev/null)
    if ! echo "$check_output" | grep -q "\.json"; then
        echo "[INFO] No label_pred or no files in latest date, will run 3dbox_extract"
        return 1
    fi
    
    echo "[INFO] Manual labels found at ${obs_label_path}, downloading..."
    rm -rf "${local_label_dir}"
    "${OBSUTIL}" sync "${obs_label_path}/" "${local_label_dir}/" -e="${ENDPOINT}"
    
    if [ $? -eq 0 ]; then
        echo "[INFO] Manual labels downloaded successfully, skipping 3dbox_extract"
        return 0
    else
        echo "[WARNING] Failed to download manual labels, will fallback to automatic extraction"
        return 1
    fi
}