#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
LINEAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINEAGE_PROJECT=${HUGSIM_ROOT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$(hugsim_source_path "$LINEAGE_PROJECT/code/HUGSIM/.pixi/envs/default/bin/python")" "$(hugsim_source_path "$LINEAGE_DIR/test_policy.py")" -v
if [[ "${1:-run}" == --check ]]; then
  exec "$(hugsim_source_path "$LINEAGE_PROJECT/code/HUGSIM/.pixi/envs/default/bin/python")" "$(hugsim_source_path "$LINEAGE_DIR/launch.py")" --check
fi
exec "$(hugsim_source_path "$LINEAGE_PROJECT/code/HUGSIM/.pixi/envs/default/bin/python")" "$(hugsim_source_path "$LINEAGE_DIR/launch.py")"
