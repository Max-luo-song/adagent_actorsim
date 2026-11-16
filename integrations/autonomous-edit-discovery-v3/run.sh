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
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase46/cutin551_white_edge_v1/native_runtime/lib")"
PY="${HUGSIM_PYTHON}"
if [ "${1:-brief}" = _probe ] || [ "${1:-brief}" = _gpu ] || [ "${1:-brief}" = _gate ] || [ "${1:-brief}" = _training_audit ]; then
 STAGE="$1";shift
 exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
 flock -n -x 9 || { echo 'GPU lock busy' >&2;exit 75; }
 test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")"
 "$PY" -B -c 'import sys;sys.path.insert(0,sys.argv[1]);from runtime_environment import gpu_idle;gpu_idle()' "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase46/cutin551_white_edge_v1")"
 if [ "$STAGE" = _training_audit ]; then exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/training_audit.py")" "$@";fi
 if [ "$STAGE" = _gate ]; then exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/gate_worker.py")" "$@";fi
 if [ "$STAGE" = _probe ]; then exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/probe_worker.py")" execute "$@";fi
 exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/flow.py")" _gpu "$@"
fi
exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/flow.py")" "${@:-brief}"
