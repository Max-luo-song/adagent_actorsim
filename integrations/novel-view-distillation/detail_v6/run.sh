#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
P26="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26")"
BRIDGE="$(hugsim_source_path "$P26/bridge")"
PY="${HUGSIM_PYTHON}"
STAGE="${1:-all}"
OUT="${2:-$P26/novel_distillation_v6/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${3:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$HERE:$HERE/..:$HERE/../quality_v2:$HERE/../conditioning_v3:$HERE/../lighting_v4:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
case "$STAGE" in
  prepare|check) exec "$PY" -u -B "$(hugsim_source_path "$HERE/experiment_v6.py")" "$STAGE" --output "$OUT";;
  all|cache|probe|train|evaluate|compare) ;;
  *) echo 'Stages: all prepare check cache probe train evaluate compare; train/evaluate require OUTPUT PROFILE' >&2; exit 2;;
esac
if [[ "$STAGE" == train || "$STAGE" == evaluate ]]; then
  test -n "$PROFILE" || { echo 'Profile required: edge_color or footprint' >&2; exit 2; }
fi
"$PY" -B -c 'import torch; assert torch.cuda.is_available(), "CUDA required; use the GPU terminal"'
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU has active processes; no work started' >&2; exit 1; fi
test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present; no work started' >&2; exit 1; }
run_stage() {
  local part="$1" name="${2:-}" suffix="${2:+_$2}"
  local arguments=("$part" --output "$OUT")
  if [[ -n "$name" ]]; then arguments+=(--profile "$name");fi
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment_v6.py")" "${arguments[@]}" 2>&1 | tee "$(hugsim_source_path "$OUT/$part$suffix.log")"
}
echo "Detail v6 output: $OUT"
if [[ "$STAGE" == all ]]; then
  "$PY" -u -B "$(hugsim_source_path "$HERE/experiment_v6.py")" prepare --output "$OUT"
  run_stage cache
  run_stage probe
  mapfile -t PROFILES < <("$PY" -B -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))["admitted_profiles"]))' "$(hugsim_source_path "$OUT/probe.json")")
  test "${#PROFILES[@]}" -gt 0
  for name in "${PROFILES[@]}";do run_stage train "$name";run_stage evaluate "$name";done
  run_stage compare
else run_stage "$STAGE" "$PROFILE";fi
