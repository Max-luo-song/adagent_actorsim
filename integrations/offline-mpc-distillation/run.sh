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
OUT="${2:-$SOURCE/resumed_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$(hugsim_source_path "$SOURCE/native_runtime/lib")"
case "$STAGE" in
 all)
  "$PY" -u -B "$(hugsim_source_path "$HERE/pipeline.py")" prepare --output "$OUT"
  for part in collect train evaluate;do
   "$PY" -u -B "$(hugsim_source_path "$HERE/pipeline.py")" "$part" --output "$OUT" 2>&1 | tee "$(hugsim_source_path "$OUT/$part.log")"
  done;;
 prepare|collect|adopt|train|evaluate|evaluate-fallback|check)
  exec "$PY" -u -B "$(hugsim_source_path "$HERE/pipeline.py")" "$STAGE" --output "$OUT";;
 *) echo 'Stages: all prepare collect adopt train evaluate evaluate-fallback check' >&2;exit 2;;
esac
