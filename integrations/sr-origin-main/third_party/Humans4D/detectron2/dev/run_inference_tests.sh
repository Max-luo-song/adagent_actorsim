#!/bin/bash -e

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Copyright (c) Facebook, Inc. and its affiliates.

BIN="python tools/train_net.py"
OUTPUT="inference_test_output"
NUM_GPUS=2

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
      OUTPUT_DIR $OUTPUT
      rm -rf $OUTPUT
done


echo "========================================================================"
echo "Running demo.py ..."
echo "========================================================================"
DEMO_BIN="python demo/demo.py"
COCO_DIR=datasets/coco/val2014
mkdir -pv $OUTPUT

set -v

$DEMO_BIN --config-file ./configs/quick_schedules/panoptic_fpn_R_50_inference_acc_test.yaml \
  --input $COCO_DIR/COCO_val2014_0000001933* --output $OUTPUT
rm -rf $OUTPUT
