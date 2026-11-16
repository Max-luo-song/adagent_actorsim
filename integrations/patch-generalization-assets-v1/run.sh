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
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGE="${1:-audit}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$STAGE" = assets ] || [ "$STAGE" = semantics ]; then
 "$PY" -B -c 'import torch;assert torch.cuda.is_available(),"CUDA unavailable; no semantic GPU job started"'
 exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
 flock -n -x 9 || { echo 'GPU lock busy; no semantic GPU job started' >&2; exit 1; }
 test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'STOP present' >&2;exit 1; }
fi
exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/pipeline.py")" "$STAGE" "$@"
