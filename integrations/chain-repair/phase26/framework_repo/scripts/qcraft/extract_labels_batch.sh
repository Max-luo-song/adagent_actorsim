
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
gpu=-1
data_root="data/qcraft/raw" # 原始数据目录
scene_list_path=data/qcraft_scenes.txt
################################################################################

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
    fi
fi
echo "Using GPU: ${gpu}"

# 激活 Conda 环境
CONDA_BASE=$(conda info --base)
source "$(hugsim_source_path "${CONDA_BASE}/etc/profile.d/conda.sh")"
conda activate sparse4d

# 获取绝对路径
abs_data_root=$(realpath "${data_root}")

cd third_party/Sparse4D
export CUDA_VISIBLE_DEVICES=${gpu}

python infer_qcraft.py \
    --data_root ${abs_data_root} \
    --scene_list_path ${scene_list_path}

cd ../../