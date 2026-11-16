
# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 参数设置
################################################################################
scene_dir="/nas/scenario_output/zhangyingjun/data/output/qcraft_20260714_141154_QCJPTDJ08789_60_80/20260731_lidar+traincam2_3_5_9_12_0_1_6_7_10_11"
edit_config_path="configs/edit_config.yaml"
output_path_postfix="_test"
render_cam_ids="0 1"
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source scripts/utils.sh

# Hardcoded camera group definitions
group0_cams=(2 3 5 9 12)
group1_cams=(0 1 6 7 10 11)

# Classify cameras and determine which groups to run
if [ -z "${render_cam_ids}" ]; then
    # Empty = render all cameras, need both groups
    groups_to_run=(0 1)
    group_cams_0=""
    group_cams_1=""
else
    group0_ids=""
    group1_ids=""
    for cam_id in ${render_cam_ids}; do
        in_group0=0
        in_group1=0
        for g0 in "${group0_cams[@]}"; do
            [ "${cam_id}" = "${g0}" ] && { in_group0=1; break; }
        done
        for g1 in "${group1_cams[@]}"; do
            [ "${cam_id}" = "${g1}" ] && { in_group1=1; break; }
        done
        if [ ${in_group0} -eq 0 ] && [ ${in_group1} -eq 0 ]; then
            echo "ERROR: Camera ${cam_id} does not belong to any training group."
            echo "  group0: 2,3,5,9,12"
            echo "  group1: 0,1,6,7,10,11"
            exit 1
        fi
        if [ ${in_group0} -eq 1 ]; then
            group0_ids="${group0_ids} ${cam_id}"
        else
            group1_ids="${group1_ids} ${cam_id}"
        fi
    done
    groups_to_run=()
    [ -n "${group0_ids}" ] && groups_to_run+=(0)
    [ -n "${group1_ids}" ] && groups_to_run+=(1)
    group_cams_0=$(echo ${group0_ids} | xargs)
    group_cams_1=$(echo ${group1_ids} | xargs)
fi

if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"

export PYTHONPATH=$(pwd)
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"

for grp in "${groups_to_run[@]}"; do
    ckpt_path="$(hugsim_source_path "${scene_dir}/group${grp}/checkpoint_final.pth")"
    if [ ! -f "${ckpt_path}" ]; then
        echo "ERROR: Checkpoint not found: ${ckpt_path}"
        exit 1
    fi
    echo "Using checkpoint: ${ckpt_path} (group${grp})"

    extra_args=""
    cams_var="group_cams_${grp}"
    if [ -n "${!cams_var}" ]; then
        extra_args="--render_cam_ids ${!cams_var}"
    fi

    python sim_render/cam/render_edit.py \
        --resume_from "${ckpt_path}" \
        --edit_config "${edit_config_path}" \
        --post_fix "${output_path_postfix}" \
        ${extra_args}
done
