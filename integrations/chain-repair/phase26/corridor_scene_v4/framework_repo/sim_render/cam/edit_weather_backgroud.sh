
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
weather_type="snowy"
output_dir="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/processed/training/20250818_134748_Q3707_200_230_part03/images_$weather_type"
data_root="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/processed/training/20250818_134748_Q3707_200_230_part03/images"
cams=(2 5 9)
################################################################################

# qcraft dataset
export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
python WeatherEdit/background_editing/src/inference.py \
        --output_dir $output_dir \
        --dataset qcraft \
        --weather_type $weather_type  \
        --data_root $data_root \
        --cams ${cams[@]} \