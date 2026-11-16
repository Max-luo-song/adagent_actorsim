#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
SCENE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ROOT=${HUGSIM_ROOT}
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SCENE_PYTHON="$(hugsim_source_path "$SCENE_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/launch.py")" --check
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/cpu_gate.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/check_wiring.py")"
exec "$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/launch.py")" "$@"
