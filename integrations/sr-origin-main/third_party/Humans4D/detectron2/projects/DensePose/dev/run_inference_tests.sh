#!/bin/bash -e

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Copyright (c) Facebook, Inc. and its affiliates.

BIN="python train_net.py"
OUTPUT="inference_test_output"
NUM_GPUS=2
IMS_PER_GPU=2
IMS_PER_BATCH=$(( NUM_GPUS * IMS_PER_GPU ))

CFG_LIST=( "${@:1}" )

if [ ${#CFG_LIST[@]} -eq 0 ]; then
  CFG_LIST=( ./configs/quick_schedules/*inference_acc_test.yaml )
fi

echo "========================================================================"
echo "Configs to run:"
echo "${CFG_LIST[@]}"
echo "========================================================================"

for cfg in "${CFG_LIST[@]}"; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN \
      --eval-only \
      --num-gpus $NUM_GPUS \
      --config-file "$cfg" \
      OUTPUT_DIR "$OUTPUT" \
      SOLVER.IMS_PER_BATCH $IMS_PER_BATCH
    rm -rf $OUTPUT
done

