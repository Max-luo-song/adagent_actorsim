#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$(hugsim_source_path "${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase46/cutin551_white_edge_v1/native_runtime/lib")"
exec "$(hugsim_source_path "${HUGSIM_ROOT}/code/HUGSIM/.pixi/envs/default/bin/python")" -u -B "$(hugsim_source_path "${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/automated_patch_workflow_v2/rounds/v15_to_v16_geometry_first/iterate.py")" "${1:-status}"
