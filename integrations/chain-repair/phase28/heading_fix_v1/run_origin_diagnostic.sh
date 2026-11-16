#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
ORIGIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIGIN_PROJECT=${HUGSIM_ROOT}
ORIGIN_PY="$(hugsim_source_path "$ORIGIN_PROJECT/code/HUGSIM/.pixi/envs/default/bin/python")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-run}" != run && "${1:-run}" != --check ]]; then
    echo 'Usage: bash run_origin_diagnostic.sh [--check]' >&2
    exit 2
fi
"$ORIGIN_PY" "$(hugsim_source_path "$ORIGIN_DIR/test_origin_diagnostics.py")" -v
if [[ "${1:-run}" == --check ]]; then
    exit 0
fi
exec "$ORIGIN_PY" "$(hugsim_source_path "$ORIGIN_DIR/run_origin_inference.py")"
