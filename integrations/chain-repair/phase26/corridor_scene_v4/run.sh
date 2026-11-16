#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P26="$(dirname "$HERE")"
ROOT="${HUGSIM_ROOT}"
BRIDGE="$(hugsim_source_path "$P26/bridge")"
CI="$(hugsim_source_path "$HERE/bridge/closed_loop_integration")"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$HERE:$CI:$HERE/bridge/framework_backend:$HERE/bridge/unified:$HERE/framework_repo:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
CHECK=0
FULL=0
OUT_OVERRIDE=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check) CHECK=1 ;;
    --full) FULL=1 ;;
    --postprocess) exec "$PY" -u -B "$(hugsim_source_path "$CI/stage_results.py")" --run "${2:?run directory required}" ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) OUT_OVERRIDE="$1" ;;
  esac
  shift
done
TASK="$(hugsim_source_path "$HERE/task.json")"
LIMIT=7200
COUNT=30
CHECK_ARGS=()
if [[ "$FULL" == 1 ]]; then TASK="$(hugsim_source_path "$HERE/full_task.json")"; LIMIT=43200; COUNT=198; CHECK_ARGS=(--full); fi
if [[ "$CHECK" == 1 ]]; then exec "$PY" -B "$(hugsim_source_path "$HERE/preflight.py")" "${CHECK_ARGS[@]}"; fi
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required: run this frozen full-scene launcher in the GPU terminal"'
"$PY" -B -c 'import os; from full_scene_contract import verify; verify(os.environ["PYTHONPATH"].split(":")[0])'
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present; no work started' >&2; exit 1; }
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU already has compute processes; no work started' >&2; exit 1; fi
OUT="${OUT_OVERRIDE:-$HERE/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$(dirname "$OUT")"
test ! -e "$OUT" && test ! -e "$OUT.log" || { echo "Refusing overwrite: $OUT" >&2; exit 1; }
printf 'Corridor v4: %s controls; wall cap %ss\nOutput: %s\n' "$COUNT" "$LIMIT" "$OUT"
"$PY" -B "$(hugsim_source_path "$HERE/launch_state.py")" running "$OUT"
set +e
timeout --signal=INT --kill-after=60s "${LIMIT}s" "$PY" -u -B "$(hugsim_source_path "$CI/run_edited_closed_loop.py")" --drive-only --task "$TASK" --output "$OUT" 2>&1 | tee "$OUT.log"
DRIVE_EXIT=${PIPESTATUS[0]}
set -e
flock -u 9
exec 9<&-
set +e
"$PY" -u -B "$(hugsim_source_path "$CI/stage_results.py")" --run "$OUT" --simulation-exit "$DRIVE_EXIT" 2>&1 | tee "$OUT.postprocess.log"
POST_EXIT=${PIPESTATUS[0]}
set -e
"$PY" -B "$(hugsim_source_path "$HERE/launch_state.py")" finished "$OUT" "$POST_EXIT"
exit "$POST_EXIT"
