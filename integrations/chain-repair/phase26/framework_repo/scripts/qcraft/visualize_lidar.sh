
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
SCENE_IDX="20251105_152839_QCOYSD504206_1240_1255"

DATA_DIR="data/qcraft/processed/training/$SCENE_IDX"
OUTPUT_DIR="output/qcraft_$SCENE_IDX"
################################################################################

export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
python datasets/qcraft/qcraft_visualize_lidar.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR