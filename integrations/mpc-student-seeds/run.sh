#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
SOURCE="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase45/policy_distill_v1")"
PY="${HUGSIM_PYTHON}"
STAGE="${1:-all}"
OUT="${2:-$SOURCE/student_seed_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$(hugsim_source_path "$SOURCE/native_runtime/lib")"
case "$STAGE" in
 all)
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" prepare --output "$OUT"
  echo "Student initial-guess experiment: $OUT"
  for step in 0 8 23;do
   VARIANTS=(baseline student)
   if [[ "$step" == 8 ]];then VARIANTS=(student baseline);fi
   for variant in "${VARIANTS[@]}";do
    "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" run --output "$OUT" --step "$step" --variant "$variant" 2>&1 | tee "$(hugsim_source_path "$OUT/${step}_${variant}.log")"
   done
  done
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" compare --output "$OUT" | tee "$(hugsim_source_path "$OUT/compare.log")";;
 prepare|compare|check) exec "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" "$STAGE" --output "$OUT";;
 run) exec "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" run --output "$OUT" --step "${3:?step required}" --variant "${4:?variant required}";;
 *) echo 'Stages: all prepare compare check run OUTPUT STEP VARIANT' >&2;exit 2;;
esac
