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
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_policy.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_depth.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_audit.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_empty.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_protocol.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_worker_io.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_speed_policy.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_current_reference.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_road_soft_cost.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_native_alignment.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/check_wiring.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_global_policy.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_agentless.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_exact_queries.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_launch_request.py")"
"$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/test_execution_reporting.py")"
exec "$SCENE_PYTHON" -B "$(hugsim_source_path "$SCENE_DIR/launch.py")" "$@"
