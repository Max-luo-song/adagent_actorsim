#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
RESUME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${HUGSIM_ROOT}
RESUME_PY="$(hugsim_source_path "$PROJECT_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")"
STATE_ROOT="$(hugsim_source_path "$RESUME_DIR/../runs/cutin551_white_edge_resumable_v1")"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH="$(hugsim_source_path "$RESUME_DIR/native_runtime/lib")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --status && "$#" == 1 ]]; then exec "$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/run_resume.py")" --status; fi
"$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/run_resume.py")" --check
if [[ "${1:-}" == --check && "$#" == 1 ]]; then "$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/validate_imports.py")"; exit 0; fi
if [[ "$#" != 0 ]]; then echo 'Usage: bash run.sh [--check|--status]'; exit 2; fi
mkdir -p "$STATE_ROOT"
exec 8>"$(hugsim_source_path "$STATE_ROOT/run.lock")"
flock -n 8 || { echo 'This resumable experiment is already running'; exit 1; }
exec 9>"$(hugsim_source_path "$PROJECT_ROOT/outputs/gpu_test_service/worker.lock")"
flock -n 9 || { echo 'GPU worker lock busy'; exit 1; }
if [[ -e "$(hugsim_source_path "$PROJECT_ROOT/outputs/gpu_test_service/STOP")" ]]; then echo 'Service STOP present'; exit 1; fi
"$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/validate_imports.py")"
"$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/run_resume.py")" --cleanup-owned --root "$STATE_ROOT"
"$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/runtime_environment.py")" --idle
timeout --signal=INT --kill-after=10s 180s "$RESUME_PY" -B "$(hugsim_source_path "$RESUME_DIR/runtime_environment.py")" --gpu
exec timeout --signal=INT --kill-after=60s 14400s "$RESUME_PY" -u -B "$(hugsim_source_path "$RESUME_DIR/run_resume.py")" --root "$STATE_ROOT"
