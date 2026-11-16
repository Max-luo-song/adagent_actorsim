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
BASE="$(hugsim_source_path "$ROOT/integrations/novel-view-distillation")"
P26="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26")"
BRIDGE="$(hugsim_source_path "$P26/bridge")"
PY="${HUGSIM_PYTHON}"
STAGE="${1:-prepare}"
OUT="${2:-$P26/novel_swap_distillation_v1/scene551_ego_native3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export LD_LIBRARY_PATH="$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase46/cutin551_white_edge_v1/native_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HERE:$BASE:$ROOT/integrations/novel-view-position-swap:$BASE/quality_v2:$BASE/conditioning_v3:$BASE/lighting_v4:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$(hugsim_source_path "$BASE/.cache/huggingface")"
case "$STAGE" in test) exec "$PY" -B -m unittest discover -s "$HERE" -p 'test_*.py' -v;; prepare|status|recover-render|recover-smpl|present) exec "$PY" -B "$(hugsim_source_path "$HERE/workflow.py")" "$STAGE" --output "$OUT";; all|render|pseudo-gt|train|evaluate) ;; *) echo 'Stages: test prepare status recover-render recover-smpl all render pseudo-gt train evaluate present' >&2;exit 2;;esac
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required; no edited rendering or training started"'
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU lock busy' >&2;exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU compute processes present' >&2;exit 1;fi
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'GPU service STOP present' >&2;exit 1; }
if [[ "$STAGE" == all ]]; then
  "$PY" -u -B "$(hugsim_source_path "$HERE/workflow.py")" prepare --output "$OUT"
  STAGES=(render pseudo-gt train evaluate present)
else STAGES=("$STAGE");fi
echo "Position-swap distillation output: $OUT"
for PART in "${STAGES[@]}";do
  CURRENT_PY="$PY"
  if [[ "$PART" == pseudo-gt ]];then CURRENT_PY=$("$PY" -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["task"]["teacher"]["python"])' "$(hugsim_source_path "$OUT/job.json")");fi
  "$CURRENT_PY" -u -B "$(hugsim_source_path "$HERE/workflow.py")" "$PART" --output "$OUT" 2>&1 | tee -a "$(hugsim_source_path "$OUT/$PART.log")"
done
