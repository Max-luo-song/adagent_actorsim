#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
LOCAL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
SOURCE="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase46/cutin551_white_edge_v1")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$(hugsim_source_path "$SOURCE/native_runtime/lib")"
case "${1:-audit}" in
 audit) exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/audit.py")";;
 prepare) exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/gpu_probe.py")" prepare;;
 probe)
  "$PY" -B -c 'import torch;assert torch.cuda.is_available(),"CUDA unavailable; no speed probe started"'
  exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
  flock -n -x 9 || { echo 'GPU worker lock busy; no speed probe started' >&2;exit 1; }
  "$PY" -B -c 'import sys;sys.path.insert(0,sys.argv[1]);from runtime_environment import gpu_idle;gpu_idle()' "$SOURCE"
  test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'STOP present; no speed probe started' >&2;exit 1; }
  exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/gpu_probe.py")" run;;
 *) echo 'Stages: audit prepare probe' >&2;exit 2;;
esac
