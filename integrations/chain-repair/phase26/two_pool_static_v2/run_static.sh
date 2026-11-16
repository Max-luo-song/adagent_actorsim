#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${HUGSIM_PYTHON}"
DEVICE="${1:-cuda}"
if [[ "$DEVICE" == cuda ]]; then
  exec 9<${HUGSIM_ROOT}/outputs/gpu_test_service/worker.lock
  if ! flock -n -x 9; then
    echo 'GPU lock busy; CPU static preparation remains running; no takeover requested.' >&2
    exit 75
  fi
  "$PY" -B -c 'import torch; assert torch.cuda.is_available(), "No CUDA device; CPU job left untouched"; print("Static semantic GPU:", torch.cuda.get_device_name(0))'
fi
exec "$PY" -u -B "$(hugsim_source_path "$D/code/pipeline.py")" --device "$DEVICE"
