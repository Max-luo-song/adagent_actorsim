#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
GLOBAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_ROOT=${HUGSIM_ROOT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
GLOBAL_PYTHON="$(hugsim_source_path "$GLOBAL_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")"
"$GLOBAL_PYTHON" -m unittest discover -s "$GLOBAL_DIR" -p 'test_*.py' -v
"$GLOBAL_PYTHON" "$(hugsim_source_path "$GLOBAL_DIR/run_harness.py")" --check-imports
exec "$GLOBAL_PYTHON" "$(hugsim_source_path "$GLOBAL_DIR/launch.py")" "$@"
