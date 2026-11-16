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
PY="${HUGSIM_PYTHON}"
P26="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26")"
BRIDGE="$(hugsim_source_path "$P26/bridge")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$HERE:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$(hugsim_source_path "$HERE/.cache/huggingface")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/novel-distill-mpl}"
STAGE="${1:-check}"
TASK="${2:-$HERE/examples/scene551_left1m/task.json}"
OUT="${3:-$P26/novel_distillation_v1/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "$STAGE" == check ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/pipeline.py")" check --task "$TASK"; fi
if [[ "$STAGE" == prepare ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/pipeline.py")" prepare --task "$TASK" --output "$OUT"; fi
case "$STAGE" in all|render|pseudo-gt|distill|evaluate|recover-evaluate) ;; *) echo 'Stages: check prepare all render pseudo-gt distill evaluate recover-evaluate' >&2; exit 2;; esac
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required; use the GPU terminal"'
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU already has compute processes; no work started' >&2; exit 1; fi
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present; no work started' >&2; exit 1; }
if [[ "$STAGE" == all ]]; then
  "$PY" -B "$(hugsim_source_path "$HERE/pipeline.py")" prepare --task "$TASK" --output "$OUT"
  STAGES=(render pseudo-gt distill evaluate)
else
  STAGES=("$STAGE")
fi
echo "Distillation output: $OUT"
for PART in "${STAGES[@]}"; do
  CURRENT_PY="$PY"
  ENTRY="$(hugsim_source_path "$HERE/pipeline.py")"
  if [[ "$PART" == recover-evaluate ]]; then ENTRY="$(hugsim_source_path "$HERE/recovery_eval_v1/evaluate.py")"; fi
  if [[ "$PART" == pseudo-gt ]]; then
    CURRENT_PY=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["task"]["teacher"]["python"])' "$(hugsim_source_path "$OUT/job.json")")
  fi
  "$CURRENT_PY" -u -B "$ENTRY" "$PART" --output "$OUT" 2>&1 | tee "$(hugsim_source_path "$OUT/$PART.log")"
done
