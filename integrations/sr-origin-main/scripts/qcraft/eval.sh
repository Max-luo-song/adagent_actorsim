
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
gpu=-1

ckpt_path="output/qcraft_20250702_133223_Q2517/20251001_lidar+cam0_1_2_3_4_5_6_8_9_10_12/checkpoint_final.pth"
################################################################################

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


echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
CUDA_VISIBLE_DEVICES=${gpu} python tools/eval.py \
    --resume_from $ckpt_path \
    # --enable_viewer