
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
scene_id="20251105_152839_QCOYSD504206_1240_1255"
label_type="gt"  # pred使用自动化标注，gt使用人工标注
################################################################################

export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
python datasets/preprocess.py \
    --data_root data/qcraft/raw/ \
    --target_dir data/qcraft/processed \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id \
    --workers 1 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar mix_novel_views\
    --skip_front_wide_side_cameras \
    --label_type ${label_type}
