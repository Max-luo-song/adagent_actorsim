#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
################################################################################
# 完整Pipeline
################################################################################

# Preprocessing reads many camera images concurrently. Raise the per-process
# file descriptor limit for this pipeline when the container permits it.
OPEN_FILES_LIMIT=65535
if ! ulimit -Sn "${OPEN_FILES_LIMIT}"; then
    echo "[WARN] Unable to raise open-file limit to ${OPEN_FILES_LIMIT}; continuing with soft=$(ulimit -Sn), hard=$(ulimit -Hn)" >&2
fi
echo "[INFO] Open-file limit: $(ulimit -Sn)"

source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
mode="${4:-1}"
label_type="${5:-pred}"
version="${6:-26.8.25.1}"
enable_difix="${7:-0}"   # 0=关闭, 1=开启 DiFix 渐进式蒸馏
novel_traj_shifts="${8:-left_1,left_2,left_3,right_1,right_2,right_3}"
novel_distill_repeat="${9:-3}"
original_distill_repeat="${10:-6}"

# 时间戳取整,start_time向下取整, end_time向上取整
clip_start_time=${clip_start_time%.*}
clip_end_time=${clip_end_time%.*}

echo "参数: scene_id=$scene_id, clip_start_time=$clip_start_time, clip_end_time=$clip_end_time, mode=$mode, label_type=$label_type, version=$version, enable_difix=$enable_difix, novel_traj_shifts=$novel_traj_shifts"
if [[ "${enable_difix}" != "1" ]]; then
    echo "[INFO] enable_difix=0, novel_traj_shifts ignored"
fi

# 避免没有权限
chmod -R 755 tools/obsutil_linux_amd64_5.8.3

# step 1: 数据下载切分和解析
echo "=========================================="
echo "✓ 开始数据下载切分和解析"
echo "=========================================="
source /opt/conda/etc/profile.d/conda.sh && conda activate base
bash scripts/qcraft/data_preparation.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$mode" "$version"
echo "=========================================="
echo "✓ 步骤结束: 数据下载切分和解析"
echo "=========================================="

# step 2: extract labels
echo "=========================================="
echo "✓ 开始标签提取"
echo "=========================================="
bash scripts/qcraft/label_extraction.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$label_type" "$version"
echo "=========================================="
echo "✓ 步骤结束: 标签提取"
echo "=========================================="

# 执行训练部分
scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
abs_data_root=$(realpath "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}")")

# step 3: preprocess
echo "=========================================="
echo "✓ 开始预处理"
echo "=========================================="
source /opt/conda/etc/profile.d/conda.sh && conda activate main
export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
python datasets/preprocess.py \
    --data_root "${abs_data_root}" \
    --target_dir "$(hugsim_source_path "${processed_data_root}/${version}")" \
    --dataset qcraft \
    --split training \
    --scene_ids "${scene_id_s_e}" \
    --workers 2 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar \
    --skip_front_wide_side_cameras \
    --label_type "${label_type}"
echo "=========================================="
echo "✓ 步骤结束: 预处理"
echo "=========================================="

# step 4: extract mask
echo "=========================================="
echo "✓ 开始掩码提取"
echo "=========================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh
source /opt/conda/etc/profile.d/conda.sh && conda activate mmseg2

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
    --data_root ${processed_data_root}/${version}/training \
    --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
    --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
    --scene_ids=${scene_id_s_e} \
    --process_dynamic_mask
echo "=========================================="
echo "✓ 步骤结束: 掩码提取"
echo "=========================================="

# step 5: process human pose
echo "=========================================="
echo "✓ 开始处理人体姿态"
echo "=========================================="
source /opt/conda/etc/profile.d/conda.sh && conda activate 4d-humans
export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root ${processed_data_root}/${version}/training \
    --dataset qcraft \
    --scene_id $scene_id_s_e
echo "=========================================="
echo "✓ 步骤结束: 处理人体姿态"
echo "=========================================="

# step 6: camera group training
echo "=========================================="
echo "✓ 开始相机分组训练"
echo "=========================================="

run_dir="$(bash "$(hugsim_source_path "${work_root}/scripts/qcraft/group_training.sh")" "${scene_id_s_e}" "${version}")" || exit 1

echo "Run directory: ${run_dir}"

check_file_exists "$(hugsim_source_path "${run_dir}/group0/checkpoint_final.pth")" "group0 checkpoint_final.pth" || exit 1
check_file_exists "$(hugsim_source_path "${run_dir}/group1/checkpoint_final.pth")" "group1 checkpoint_final.pth" || exit 1

echo "=========================================="
echo "✓ 步骤结束: 相机分组训练"
echo "=========================================="

# step 7: lidar训练
echo "=========================================="
echo "✓ 开始LiDAR训练"
echo "=========================================="
source /opt/conda/etc/profile.d/conda.sh && conda activate main
LIDAR_PROJECT_ROOT="$(hugsim_source_path "${work_root}/lidargs")"
echo "LiDAR project root: ${LIDAR_PROJECT_ROOT}"
export PYTHONPATH="${LIDAR_PROJECT_ROOT}:${PYTHONPATH}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"

echo "Data path: ${processed_data_root}/${version}/training/${scene_id_s_e}"

lidar_cpt_path="$(hugsim_source_path "${run_dir}/lidar_cpt")"

mkdir -p "$lidar_cpt_path"
echo "LiDAR checkpoint path: ${lidar_cpt_path}"

python ${LIDAR_PROJECT_ROOT}/train.py \
    -s ${processed_data_root}/${version}/training/${scene_id_s_e} \
    -m ${lidar_cpt_path} \
    --caseid ${scene_id_s_e} \
    --gpu ${gpu} \
    --iterations ${train_lidar_iterations} \
    --max_depth ${max_depth} \
    --dataset chery \
    --block_size ${block_size}
check_dir_not_empty "$lidar_cpt_path" "lidar_cpt" || exit 1

# step 8: DiFix progressive distillation（enable_difix=1 时执行）
if [[ "${enable_difix}" == "1" ]]; then
    echo "=========================================="
    echo "✓ 开始 DiFix 渐进式蒸馏（分组串行）"
    echo "  novel_traj_shifts: ${novel_traj_shifts}"
    echo "  output_dir: ${run_dir}"
    echo "=========================================="

    group_0_difix_ckpt_path="$(hugsim_source_path "${run_dir}/group0/checkpoint_final.pth")"
    group_1_difix_ckpt_path="$(hugsim_source_path "${run_dir}/group1/checkpoint_final.pth")"
    group_0_original_view_ckpt_path="$(hugsim_source_path "${run_dir}/group0/checkpoint_final_original_view.pth")"
    group_1_original_view_ckpt_path="$(hugsim_source_path "${run_dir}/group1/checkpoint_final_original_view.pth")"

    # If an original-view backup exists, restore it before DiFix so repeated runs
    # always start from the non-distilled model.
    if [ -f "${group_0_original_view_ckpt_path}" ]; then
        echo "[INFO] Restoring group0 original-view checkpoint to checkpoint_final.pth"
        rm -f "${group_0_difix_ckpt_path}" || exit 1
        cp -p "${group_0_original_view_ckpt_path}" "${group_0_difix_ckpt_path}" || exit 1
    fi
    if [ -f "${group_1_original_view_ckpt_path}" ]; then
        echo "[INFO] Restoring group1 original-view checkpoint to checkpoint_final.pth"
        rm -f "${group_1_difix_ckpt_path}" || exit 1
        cp -p "${group_1_original_view_ckpt_path}" "${group_1_difix_ckpt_path}" || exit 1
    fi

    if [ ! -f "${group_0_difix_ckpt_path}" ] || [ ! -f "${group_1_difix_ckpt_path}" ]; then
        echo "[ERROR] group0 or group1 3DGS checkpoints not found"
        echo "  expected: ${group_0_difix_ckpt_path}"
        echo "  expected: ${group_1_difix_ckpt_path}"
        exit 1
    fi

    # 与 DiFix 脚本使用相同的规则，提前确定蒸馏产物路径。
    source scripts/qcraft/difix_schedule.sh
    difix_ckpt_suffix=$(build_difix_ckpt_suffix \
        "${novel_traj_shifts}" \
        "${novel_distill_repeat}" \
        "${original_distill_repeat}") || exit 1
    group_0_difix_final_ckpt_path="$(hugsim_source_path "${run_dir}/group0/checkpoint_final_${difix_ckpt_suffix}.pth")"
    group_1_difix_final_ckpt_path="$(hugsim_source_path "${run_dir}/group1/checkpoint_final_${difix_ckpt_suffix}.pth")"

    source /opt/conda/etc/profile.d/conda.sh && conda activate main
    cd "${work_root}"
    export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"

    # group 0 difix training
    env \
        ckpt_path="${group_0_difix_ckpt_path}" \
        experiment_output_dir="$(hugsim_source_path "${run_dir}/group0")" \
        difix_group=group0 \
        novel_traj_shifts="${novel_traj_shifts}" \
        novel_distill_repeat="${novel_distill_repeat}" \
        original_distill_repeat="${original_distill_repeat}" \
        pipeline_gpu="${gpu}" \
        bash sim_render/cam/render_novel_trajectory_with_difix.sh

    if [ $? -ne 0 ]; then
        echo "[ERROR] group 0 DiFix distillation failed"
        exit 1
    fi

    # group 1 difix training
    env \
        ckpt_path="${group_1_difix_ckpt_path}" \
        experiment_output_dir="$(hugsim_source_path "${run_dir}/group1")" \
        difix_group=group1 \
        novel_traj_shifts="${novel_traj_shifts}" \
        novel_distill_repeat="${novel_distill_repeat}" \
        original_distill_repeat="${original_distill_repeat}" \
        pipeline_gpu="${gpu}" \
        bash sim_render/cam/render_novel_trajectory_with_difix.sh

    if [ $? -ne 0 ]; then
        echo "[ERROR] group 1 DiFix distillation failed"
        exit 1
    fi

    if [ ! -f "${group_0_difix_final_ckpt_path}" ] || [ ! -f "${group_1_difix_final_ckpt_path}" ]; then
        echo "[ERROR] DiFix final checkpoint not found:"
        echo "  ${group_0_difix_final_ckpt_path}"
        echo "  ${group_1_difix_final_ckpt_path}"
        exit 1
    fi

    if [ ! -f "${group_0_original_view_ckpt_path}" ]; then
        cp -p "${group_0_difix_ckpt_path}" "${group_0_original_view_ckpt_path}" || exit 1
        echo "[INFO] Original checkpoint backed up to: ${group_0_original_view_ckpt_path}"
    else
        echo "[INFO] Keep existing original-view checkpoint: ${group_0_original_view_ckpt_path}"
    fi
    if [ ! -f "${group_1_original_view_ckpt_path}" ]; then
        cp -p "${group_1_difix_ckpt_path}" "${group_1_original_view_ckpt_path}" || exit 1
        echo "[INFO] Original checkpoint backed up to: ${group_1_original_view_ckpt_path}"
    else
        echo "[INFO] Keep existing original-view checkpoint: ${group_1_original_view_ckpt_path}"
    fi
    rm -f "${group_0_difix_ckpt_path}" || exit 1
    cp -f "${group_0_difix_final_ckpt_path}" "${group_0_difix_ckpt_path}" || exit 1
    rm -f "${group_1_difix_ckpt_path}" || exit 1
    cp -f "${group_1_difix_final_ckpt_path}" "${group_1_difix_ckpt_path}" || exit 1
    echo "[INFO] DiFix checkpoint promoted to: ${group_0_difix_ckpt_path}"
    echo "[INFO] DiFix checkpoint promoted to: ${group_1_difix_ckpt_path}"

    echo "=========================================="
    echo "✓ 步骤结束: DiFix 渐进式蒸馏"
    echo "=========================================="
fi

# step 9: 上传数据
echo "=========================================="
echo "✓ 开始上传数据"
echo "=========================================="
bash scripts/qcraft/upload.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$version" "$run_dir"
