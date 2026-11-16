#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
SCENARIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${HUGSIM_ROOT}
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH="$(hugsim_source_path "$SCENARIO_DIR/native_runtime/lib")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SCENARIO_PY="$(hugsim_source_path "$PROJECT_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")"
"$SCENARIO_PY" -B "$(hugsim_source_path "$SCENARIO_DIR/run_cutin.py")" --check
if [[ "${1:-}" == --check && "$#" == 1 ]]; then exit 0; fi
if [[ "$#" != 0 ]]; then echo "Usage: bash run.sh [--check]"; exit 2; fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable; no simulation launched"; exit 1; }
"$SCENARIO_PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; no simulation launched"'
exec 9>"$(hugsim_source_path "$PROJECT_ROOT/outputs/gpu_test_service/worker.lock")"
flock -n 9 || { echo 'GPU worker lock busy'; exit 1; }
if [[ -e "$(hugsim_source_path "$PROJECT_ROOT/outputs/gpu_test_service/STOP")" ]]; then echo 'Service STOP present'; exit 1; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q '^[[:space:]]*[0-9]'; then echo 'GPU compute process active'; exit 1; fi
SCENARIO_OUTPUT="$(hugsim_source_path "$SCENARIO_DIR/../runs/cutin551_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}")"
mkdir -p "$(dirname "$SCENARIO_OUTPUT")"
set +e
timeout --signal=INT --kill-after=60s 14400s "$SCENARIO_PY" -u -B "$(hugsim_source_path "$SCENARIO_DIR/run_cutin.py")" --output "$SCENARIO_OUTPUT" 2>&1 | tee "$SCENARIO_OUTPUT.simulation.log"
SIM_EXIT=${PIPESTATUS[0]}
set -e
flock -u 9
exec 9>&-
POST_EXIT=0
if [[ -f "$(hugsim_source_path "$SCENARIO_OUTPUT/run_report.json")" ]]; then
 "$SCENARIO_PY" -B "$(hugsim_source_path "$SCENARIO_DIR/postprocess.py")" --run "$SCENARIO_OUTPUT" || POST_EXIT=$?
fi
if [[ "$SIM_EXIT" != 0 ]]; then exit "$SIM_EXIT"; fi
exit "$POST_EXIT"
