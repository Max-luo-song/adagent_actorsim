#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 通用 ego mask 替换脚本
# 支持两种类型：render（渲染用）和 train（训练用）
#
# 用法:
#   bash replace_egomask.sh <mask_type> <scene_id> <target_dir> [OBS_BUCKET] [OBS_BASE_PATH] [OBSUTIL] [ENDPOINT] [OBS_FILTER]
#
# 参数说明:
#   mask_type       : 必须为 "render" 或 "train"
#   scene_id        : 场景 ID，例如 "xxx_yyy_zzz"，不管有没有后缀_start_end都可以
#   target_dir      : 目标本地目录，下载的 mask 文件夹将临时放在这里，最终合并到其下的 ego_masks/
#   OBS_BUCKET      : 可选，默认 obs://hwcn-hd2-ad-simulation
#   OBS_BASE_PATH   : 可选，默认 scene_reconstruction
#   OBSUTIL         : 可选，默认 tools/obsutil_linux_amd64_5.8.3/obsutil
#   ENDPOINT        : 可选，默认 obs.cn-east-4.myhuaweicloud.com
#   OBS_FILTER      : 可选，默认 '\[[=>_ \-]+\]|[0-9]+\.[0-9]+%|tps:|Succeed count|Failed count|Succeed bytes|Metrics|Task id| [0-9]+/[0-9]+ |[0-9]+\.[0-9]+KB/s|[0-9]+\.[0-9]+MB/[0-9]+\.[0-9]+MB'

set -u

# ==================== 参数解析 ====================
mask_type="${1:?Usage: bash $0 <mask_type> <scene_id> <target_dir> [OBS_BUCKET] [OBS_BASE_PATH] [OBSUTIL] [ENDPOINT] [OBS_FILTER]}"
scene_id="${2:?Missing scene_id}"
target_dir="${3:?Missing target_dir}"

# OBS 配置（带默认值，可按需传入覆盖）
OBS_BUCKET="${4:-obs://hwcn-hd2-ad-simulation}"
OBS_BASE_PATH="${5:-scene_reconstruction}"
OBSUTIL="${6:-tools/obsutil_linux_amd64_5.8.3/obsutil}"
ENDPOINT="${7:-obs.cn-east-4.myhuaweicloud.com}"
# 过滤 OBS 输出中的无用进度信息
DEFAULT_OBS_FILTER='\[[=>_ \-]+\]|[0-9]+\.[0-9]+%|tps:|Succeed count|Failed count|Succeed bytes|Metrics|Task id| [0-9]+/[0-9]+ |[0-9]+\.[0-9]+KB/s|[0-9]+\.[0-9]+MB/[0-9]+\.[0-9]+MB'
OBS_FILTER="${8:-$DEFAULT_OBS_FILTER}"

# ==================== 下载函数（带重试） ====================
download_with_retry() {
    local obs_path="$1"
    local local_path="$2"
    local description="$3"
    local max_retries=3
    local retry_count=0
    local success=0

    echo "[INFO] Downloading ${description}..."
    echo "[INFO] From: ${obs_path}"
    echo "[INFO] To: ${local_path}"

    while [ $retry_count -lt $max_retries ] && [ $success -eq 0 ]; do
        if [ $retry_count -gt 0 ]; then
            echo "[WARN] Retry attempt $retry_count/$max_retries for ${description}..."
            sleep $((retry_count * 5))
        fi

        mkdir -p "$(dirname ${local_path})" 2>/dev/null || true

        ${OBSUTIL} cp "${obs_path}" "${local_path}" -r -f \
            -e="${ENDPOINT}" 2>&1 | grep -vE "${OBS_FILTER}"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "[INFO] ${description} download success on attempt $((retry_count+1))"
            success=1
        else
            echo "[ERROR] ${description} download failed on attempt $((retry_count+1))"
            retry_count=$((retry_count + 1))
            # rm -rf "${local_path}" 2>/dev/null || true
        fi
    done

    if [ $success -eq 0 ]; then
        echo "[ERROR] ${description} download failed after $max_retries attempts"
        return 1
    fi
    return 0
}

# ==================== 根据 mask_type 设置变量 ====================
scene_suffix=$(echo "${scene_id}" | cut -d'_' -f3)

case "${mask_type}" in
    render)
        mask_folder_name="render_ego_masks"
        probe_file="0.png"
        ;;
    train)
        mask_folder_name="train_ego_masks"
        probe_file="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110.png"
        ;;
    *)
        echo "[ERROR] Invalid mask_type: ${mask_type} (must be 'render' or 'train')"
        exit 1
        ;;
esac

OBS_MASK_ROOT="$(hugsim_source_path "${OBS_BUCKET}/${OBS_BASE_PATH}/ego_masks")"
NEW_MASK_OBS="$(hugsim_source_path "${OBS_MASK_ROOT}/${scene_suffix}/${mask_folder_name}")"
DEFAULT_MASK_OBS="$(hugsim_source_path "${OBS_MASK_ROOT}/template/${mask_folder_name}")"

# ==================== 下载并替换 ====================
if ${OBSUTIL} stat "$(hugsim_source_path "${NEW_MASK_OBS}/${probe_file}")" -e="${ENDPOINT}" >/dev/null 2>&1; then
    echo "[INFO] Found custom ${mask_type} masks for ${scene_id}, downloading..."
    download_with_retry \
        "${NEW_MASK_OBS}/" \
        "${target_dir}/" \
        "${mask_type} ego_masks for ${scene_id}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to download ${mask_type} masks"
        exit 1
    fi
    echo "[INFO] Downloaded custom ${mask_type} masks from ${NEW_MASK_OBS}"
else
    echo "[INFO] No custom ${mask_type} masks for suffix ${scene_suffix}, using default masks..."
    download_with_retry \
        "${DEFAULT_MASK_OBS}/" \
        "${target_dir}/" \
        "default ${mask_type} ego_masks for ${scene_id}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to download default ${mask_type} masks"
        exit 1
    fi
    echo "[INFO] Downloaded default ${mask_type} masks from ${DEFAULT_MASK_OBS}"
fi

# 由于 OBS 的 cp 会把整个 mask 文件夹（如 render_ego_masks/）下载到目标目录，
# 需将其内容合并到 ego_masks/ 下，并删除临时文件夹，注意避免重命名这种要同时删除和创建的原子操作可能导致的权限问题。
# 使用 cp -rf 覆盖，无论目标 ego_masks 是否存在都能正确处理。
cp -rf "$(hugsim_source_path "${target_dir}/${mask_folder_name}/.")" "$(hugsim_source_path "${target_dir}/ego_masks")"
rm -rf "$(hugsim_source_path "${target_dir}/${mask_folder_name}")"

echo "[INFO] Ego masks replacement completed for ${mask_type}."
exit 0