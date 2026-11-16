#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Run on the user's Inspire server. Existing HUGSIM environment only.
set -eu
BASE=${HUGSIM_ROOT}
TOOLS="$BASE/waymo23-native-import"
INPUTS="$BASE/waymo23-inputs"
CHECKPOINT=${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_23/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth
ARCHIVE="$BASE/../uploads/waymo23-adapter-inputs.tar.gz"
PIXI="$BASE/tools/pixi/bin/pixi"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
RUN="$BASE/outputs/waymo23_import_$RUN_TAG"
if [ ! -f "$INPUTS/config.yaml" ]; then
  mkdir -p "$INPUTS"
  tar -xzf "$ARCHIVE" -C "$INPUTS"
fi
test -s "$CHECKPOINT"
mkdir -p "$RUN"
cd "$BASE/code/HUGSIM"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "Source checkpoint: $CHECKPOINT"
echo "Output: $RUN"
"$PIXI" run python "$(hugsim_source_path "$TOOLS/export_snapshot.py")" \
  --checkpoint "$CHECKPOINT" --inputs "$INPUTS" \
  --output "$RUN/native_snapshot" --frame 0 --camera 0 --scale .5 \
  > "$RUN/export.log" 2>&1 || { cat "$RUN/export.log"; exit 1; }
cat "$RUN/export.log"
"$PIXI" run python "$(hugsim_source_path "$TOOLS/render_native.py")" \
  --hugsim "$BASE/code/HUGSIM" --package "$RUN/native_snapshot" \
  --output "$RUN/render" > "$RUN/render.log" 2>&1 || { cat "$RUN/render.log"; exit 1; }
cat "$RUN/render.log"
echo "Inspect image: $RUN/render/rgb.png"
echo 'This checks one imported view. Dynamic closed loop and HD-score are NOT validated.'
