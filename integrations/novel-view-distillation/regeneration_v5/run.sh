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
STAGE="${1:-prepare}"
OUT="${2:-$P26/novel_distillation_v5/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$HERE:$HERE/..:$HERE/../quality_v2:$HERE/../conditioning_v3:$HERE/../lighting_v4:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$(hugsim_source_path "$HERE/../.cache/huggingface")"
if [[ "$STAGE" == prepare ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/workflow.py")" prepare --output "$OUT"; fi
case "$STAGE" in all|render|pseudo-gt|train|evaluate) ;; *) echo 'Stages: prepare all render pseudo-gt train evaluate' >&2; exit 2;; esac
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required; use the GPU terminal"'
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU has active processes; no work started' >&2; exit 1; fi
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present; no work started' >&2; exit 1; }
if [[ "$STAGE" == all ]]; then
  "$PY" -u -B "$(hugsim_source_path "$HERE/workflow.py")" prepare --output "$OUT"
  STAGES=(render pseudo-gt train evaluate)
else STAGES=("$STAGE");fi
echo "Conditioned v5 output: $OUT"
for PART in "${STAGES[@]}"; do
  CURRENT_PY="$PY"
  if [[ "$PART" == pseudo-gt ]]; then
    CURRENT_PY=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["task"]["teacher"]["python"])' "$(hugsim_source_path "$OUT/job.json")")
  fi
  "$CURRENT_PY" -u -B "$(hugsim_source_path "$HERE/workflow.py")" "$PART" --output "$OUT" 2>&1 | tee "$(hugsim_source_path "$OUT/$PART.log")"
done
