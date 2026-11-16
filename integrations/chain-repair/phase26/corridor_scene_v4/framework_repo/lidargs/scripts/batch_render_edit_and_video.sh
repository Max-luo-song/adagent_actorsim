#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
trap 'exit 1' INT
# batch_render_edit_and_video.sh — 批量 LiDAR 渲染（默认 120 前向）
# 用法: bash lidargs/scripts/batch_render_edit_and_video.sh
#
# 替代旧的 renderedit.sh/renderedit.py（已删除，能力已同步进 render.py）。
# 纯 shell，无中间 py 层：每个 clip 依次跑
#   1) sim_render/lidar/render.py    —— 渲染，落盘逐帧 pcd + ego_positions.json
#   2) utils/lidar_edit_video.py     —— 读结果画 BEV mp4（no_video=1 时跳过）
# 两者都是独立可单独调用的命令行工具。

# 【必填】Clip 列表 / 源数据路径 / YAML / 输出子目录名，四者一一对应。
# 输出规则：不写进 model_path（只读 checkpoint），写到其所在场景 output
# 目录下新建的 edit_lidar/<RUN_NAME>/，跟 lidar_cpt/ 平级；换 RUN_NAME
# 即可让多次编辑各自独立、不互相覆盖。

CLIPS=(
    "/nas/scenario_output/zhangyingjun/data/output/qcraft_20260714_141154_QCJPTDJ08789_60_80/20260731_lidar+traincam2_3_5_9_12_0_1_6_7_10_11/lidar_cpt"
)

SOURCE_PATHS=(
    "/nas/scenario_output/zhangyingjun/data/processed/training/20260714_141154_QCJPTDJ08789_60_80"
)

YAMLS=(
    "/nas/oldbak/yx/lidarM8save/configs/edit_config_qcjptdj08789_oncoming_fullscene.yaml"   # 仅 enable_edit=1 时使用
)

RUN_NAMES=(
    "oncoming_fullscene"   # 输出落在 {model_path 所在 output 目录}/edit_lidar/${RUN_NAMES[i]}/
)

# 【必填】集中日志目录
LOG_DIR="/nas/oldbak/yx/lidarM8save/batch_logs"

# 公共参数
block_size=50
iteration=-1
max_depth=100

enable_edit=1    # 1=启用编辑(传 --edit_yaml)  0=纯渲染(不传，读 checkpoint 自带标注)
no_video=0       # 1=跳过 BEV 视频生成，只跑 render.py
mode="120"       # 120: HFOV=120° → dataset=chery  |  360: HFOV=360° → dataset=chery_lidar_360

# BEV 视频参数（仅 no_video=0 时用）
video_fps=10
bev_width=1200
bev_height=900
bev_ymin=-30
bev_ymax=30
bev_point_size=0.2
# 120: x 轴 (-10, 50)，自车在图像下方约 17%，前方视野更宽
# 360: 不传 xmin/xmax，用默认值 (-30, 30)
if [ "${mode}" = "120" ]; then
    bev_xmin=-10
    bev_xmax=50
else
    bev_xmin=""
    bev_xmax=""
fi

# 最大并行 job 数（1=串行）
MAX_PARALLEL=1

# ------------------------------------------------------------------------------

if [ "${#CLIPS[@]}" -ne "${#YAMLS[@]}" ] || [ "${#CLIPS[@]}" -ne "${#SOURCE_PATHS[@]}" ] || [ "${#CLIPS[@]}" -ne "${#RUN_NAMES[@]}" ]; then
    echo "[ERROR] CLIPS / SOURCE_PATHS / YAMLS / RUN_NAMES 数量不一致 (${#CLIPS[@]} / ${#SOURCE_PATHS[@]} / ${#YAMLS[@]} / ${#RUN_NAMES[@]})，退出。"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lidargs:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"

if [ "${mode}" = "360" ]; then
    dataset="chery_lidar_360"   # 360° 全景, hfov=2π
elif [ "${mode}" = "120" ]; then
    dataset="chery"             # 120° 前向, hfov=2π/3
else
    echo "[ERROR] mode must be '120' or '360', got: ${mode}"; exit 1
fi

mkdir -p "${LOG_DIR}"

TOTAL="${#CLIPS[@]}"
BATCH_START="$(date '+%Y-%m-%d %H:%M:%S')"
MASTER_LOG="$(hugsim_source_path "${LOG_DIR}/batch_master.log")"

{
echo "============================================================"
echo "  LiDAR Batch Edit Renderer  —  共 ${TOTAL} 个 clip"
echo "  并行数  : ${MAX_PARALLEL}"
echo "  开始时间: ${BATCH_START}"
echo "  日志目录: ${LOG_DIR}"
echo "============================================================"
} | tee "${MASTER_LOG}"

run_one() {
    local idx="$1"
    local model_path="${CLIPS[$idx]}"
    local source_path="${SOURCE_PATHS[$idx]}"
    local edit_config="${YAMLS[$idx]}"
    local run_name="${RUN_NAMES[$idx]}"
    # model_path 是原始训练 checkpoint（如 .../lidar_cpt），只读，不能写进去。输出
    # 落到它所在场景 output 目录下新建的 edit_lidar/<run_name>/，跟 lidar_cpt/ 平级，
    # 不同 run_name 各自独立目录，多次编辑不会互相覆盖。
    local scene_output_root
    scene_output_root="$(dirname "${model_path}")"
    local output_path="$(hugsim_source_path "${scene_output_root}/edit_lidar/${run_name}")"
    local render_dir="$(hugsim_source_path "${output_path}/renders_edit")"
    local video_dir="$(hugsim_source_path "${output_path}/rendervideo")"

    local log_file="$(hugsim_source_path "${LOG_DIR}/render_${run_name}.log")"

    mkdir -p "${output_path}"

    {
    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(( idx + 1 ))/${TOTAL}] 开始: $(date '+%H:%M:%S')"
    echo "  model_path : ${model_path}"
    echo "  source_path: ${source_path}"
    echo "  edit_config: ${edit_config}"
    echo "  run_name   : ${run_name}"
    echo "  output_path: ${output_path}"
    echo "  log        : ${log_file}"
    echo "------------------------------------------------------------"
    } | tee -a "${MASTER_LOG}"

    if [ ! -d "${model_path}" ]; then
        echo "  [SKIP] model_path 不存在: ${model_path}" | tee -a "${MASTER_LOG}"
        return
    fi

    if [ "${enable_edit}" = "1" ] && [ ! -f "${edit_config}" ]; then
        echo "  [WARN] edit_config 不存在: ${edit_config}，跳过本 clip" | tee -a "${MASTER_LOG}"
        return
    fi

    # 第一步：渲染
    RENDER_CMD="python3 ${PROJECT_ROOT}/sim_render/lidar/render.py \
        -m \"${model_path}\" \
        --source_path \"${source_path}\" \
        --output_path \"${output_path}\" \
        --dataset ${dataset} \
        --block_size ${block_size} \
        --iteration ${iteration} \
        --max_depth ${max_depth}"
    [ "${enable_edit}" = "1" ] && RENDER_CMD="${RENDER_CMD} --edit_yaml \"${edit_config}\""

    : > "${log_file}"
    eval ${RENDER_CMD} 2>&1 | tee -a "${log_file}"
    local render_exit="${PIPESTATUS[0]}"

    if [ "${render_exit}" -ne 0 ]; then
        echo "  [ERR] clip $(( idx + 1 )) 渲染失败 (exit=${render_exit})，跳过视频: $(date '+%H:%M:%S')  → ${log_file}" | tee -a "${MASTER_LOG}"
        return
    fi

    # 第二步：BEV 视频（no_video=1 时跳过）
    local video_exit=0
    if [ "${no_video}" != "1" ]; then
        mkdir -p "${video_dir}"
        VIDEO_CMD="python3 ${PROJECT_ROOT}/utils/lidar_edit_video.py \
            --render_dir \"${render_dir}\" \
            --out_dir \"${video_dir}\" \
            --lidar_id 0 \
            --video_fps ${video_fps} \
            --bev_width ${bev_width} \
            --bev_height ${bev_height} \
            --bev_ymin ${bev_ymin} \
            --bev_ymax ${bev_ymax} \
            --bev_point_size ${bev_point_size}"
        [ -n "${bev_xmin}" ] && VIDEO_CMD="${VIDEO_CMD} --bev_xmin ${bev_xmin} --bev_xmax ${bev_xmax}"

        eval ${VIDEO_CMD} 2>&1 | tee -a "${log_file}"
        video_exit="${PIPESTATUS[0]}"
    fi

    if [ "${video_exit}" -eq 0 ]; then
        echo "  [OK ] clip $(( idx + 1 )) 完成: $(date '+%H:%M:%S')  → ${log_file}" | tee -a "${MASTER_LOG}"
    else
        echo "  [ERR] clip $(( idx + 1 )) 视频生成失败 (exit=${video_exit}): $(date '+%H:%M:%S')  → ${log_file}" | tee -a "${MASTER_LOG}"
    fi
}

job_count=0
for i in "${!CLIPS[@]}"; do
    if [ "${MAX_PARALLEL}" -gt 1 ]; then
        run_one "$i" &
        job_count=$(( job_count + 1 ))
        if [ "${job_count}" -ge "${MAX_PARALLEL}" ]; then
            wait
            job_count=0
        fi
    else
        run_one "$i"
    fi
done

wait

{
echo ""
echo "============================================================"
echo "  全部 ${TOTAL} 个 clip 处理完毕: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志汇总: ${MASTER_LOG}"
echo "  各 clip 详细日志: ${LOG_DIR}/render_*.log"
echo "============================================================"
} | tee -a "${MASTER_LOG}"
