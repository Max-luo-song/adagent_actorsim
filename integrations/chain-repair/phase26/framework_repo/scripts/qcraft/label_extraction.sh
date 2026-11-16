#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
################################################################################
# 标签提取
################################################################################
source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
label_type="${4:-pred}"
version="${5:-26.5.20.1}"

echo "参数: scene_id=$scene_id, clip_start_time=$clip_start_time, clip_end_time=$clip_end_time, label_type=$label_type, version=$version"

scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}

if [ "${label_type}" = "pred" ]; then
    echo "Step 0: Extracting labels with Sparse4D..."
    source /opt/conda/etc/profile.d/conda.sh && conda activate sparse4d
    export LD_LIBRARY_PATH=/opt/conda/envs/sparse4d/lib:$LD_LIBRARY_PATH

    ln -s /opt/conda/envs/sparse4d/lib/python3.9/site-packages/deformable_aggregation_ext.cpython-39-x86_64-linux-gnu.so third_party/Sparse4D/projects/mmdet3d_plugin/ops

    # cd third_party/Sparse4D
    # CUDA_VISIBLE_DEVICES=${gpu} python infer_qcraft.py \
    #     --data_root ${abs_data_root} \
    #     --scene_idx ${scene_id_s_e}

    # cd ../../
else
    echo "Step 0: Skipping label extraction, using GT labels..."
    source /opt/conda/etc/profile.d/conda.sh && conda activate sparse4d
    export LD_LIBRARY_PATH=/opt/conda/envs/sparse4d/lib:$LD_LIBRARY_PATH

    ln -s /opt/conda/envs/sparse4d/lib/python3.9/site-packages/deformable_aggregation_ext.cpython-39-x86_64-linux-gnu.so third_party/Sparse4D/projects/mmdet3d_plugin/ops

    parsed_label_path="$(hugsim_source_path "${parsed_data_root}/${scene_id}/${scene_id_s_e}/")"
    "${OBSUTIL}" cp \
        "obs://hwcn-hd2-ad-simulation/scene_reconstruction/${scene_id_s_e}/${version}/parsed/label"\
        "${parsed_label_path}" \
        -r -f \
        -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true
fi