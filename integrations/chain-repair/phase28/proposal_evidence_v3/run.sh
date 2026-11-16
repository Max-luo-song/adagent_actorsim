#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
EVIDENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_ROOT=${HUGSIM_ROOT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$(hugsim_source_path "$EVIDENCE_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")" -m unittest discover -s "$EVIDENCE_DIR" -p 'test_*.py' -v
exec "$(hugsim_source_path "$EVIDENCE_ROOT/code/HUGSIM/.pixi/envs/default/bin/python")" "$(hugsim_source_path "$EVIDENCE_DIR/run_inference.py")" "$@"
