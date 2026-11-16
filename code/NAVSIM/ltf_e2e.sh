#!/usr/bin/zsh

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot


NAVSIM_PATH=${HUGSIM_ROOT}/code/NAVSIM
cd ${NAVSIM_PATH}
echo ${PWD}
CHECKPOINT=${LTF_CHECKPOINT:-./ckpts/ltf_seed_0.ckpt}
CUDA_VISIBLE_DEVICES=${1} pixi run python ltf_e2e.py   output=$2   agent.checkpoint_path=$CHECKPOINT
cd -
