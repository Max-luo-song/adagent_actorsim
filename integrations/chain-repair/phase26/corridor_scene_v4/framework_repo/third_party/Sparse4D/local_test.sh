
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
export PYTHONPATH=$PYTHONPATH:./
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
export CUDA_VISIBLE_DEVICES=3
export PORT=29532

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}

cfg=sparse4dv3_temporal_r50_1x8_bs6_256x704

config=projects/configs/${cfg}.py
checkpoint=ckpt/sparse4dv3_r50.pth

echo "number of gpus: "${gpu_num}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_test.sh \
        ${config} \
        ${checkpoint} \
        ${gpu_num} \
        --eval bbox \
        $@
else
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --eval bbox \
        $@
fi
