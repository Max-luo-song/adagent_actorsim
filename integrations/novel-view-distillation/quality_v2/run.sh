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
P26="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26")"
BRIDGE="$(hugsim_source_path "$P26/bridge")"
PY="${HUGSIM_PYTHON}"
STAGE="${1:-check}"
OUT="${2:-$P26/novel_distillation_v2/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${3:-all}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$HERE:$HERE/..:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
if [[ "$STAGE" == check ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/experiment.py")" check; fi
if [[ "$STAGE" == prepare ]]; then
  ARGS=()
  if [[ "$PROFILE" != all ]]; then ARGS=(--profiles "$PROFILE"); fi
  exec "$PY" -B "$(hugsim_source_path "$HERE/experiment.py")" prepare --output "$OUT" "${ARGS[@]}"
fi
if [[ "$STAGE" == compare ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/experiment.py")" compare --output "$OUT"; fi
case "$STAGE" in all|train|evaluate) ;; *) echo 'Stages: check prepare all train evaluate compare' >&2; exit 2;; esac
if [[ "$STAGE" != all && "$PROFILE" == all ]]; then echo 'train/evaluate require a profile as the third argument' >&2; exit 2; fi
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required; use the GPU terminal"'
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU has active compute processes; no work started' >&2; exit 1; fi
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present; no work started' >&2; exit 1; }
if [[ "$STAGE" == all ]]; then
  ARGS=()
  if [[ "$PROFILE" != all ]]; then ARGS=(--profiles "$PROFILE"); fi
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" prepare --output "$OUT" "${ARGS[@]}"
  PROFILES=(lr_probe balanced structure)
  if [[ "$PROFILE" != all ]]; then PROFILES=("$PROFILE"); fi
  echo "Quality v2 output: $OUT"
  for NAME in "${PROFILES[@]}"; do
    "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" train --output "$OUT" --profile "$NAME" 2>&1 | tee "$(hugsim_source_path "$OUT/$NAME.train.log")"
    "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" evaluate --output "$OUT" --profile "$NAME" 2>&1 | tee "$(hugsim_source_path "$OUT/$NAME.evaluate.log")"
  done
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" compare --output "$OUT" 2>&1 | tee "$(hugsim_source_path "$OUT/comparison.log")"
else
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment.py")" "$STAGE" --output "$OUT" --profile "$PROFILE" 2>&1 | tee "$(hugsim_source_path "$OUT/$PROFILE.$STAGE.log")"
fi
