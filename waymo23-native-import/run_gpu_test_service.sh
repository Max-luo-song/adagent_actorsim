#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$BASE"
mkdir -p outputs/gpu_test_service
if [[ -f outputs/gpu_test_service/STOP ]]; then
    echo "GPU test worker is stopped by outputs/gpu_test_service/STOP; inspect that file before restarting." >&2
    exit 1
fi
exec env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$BASE/code/HUGSIM/.pixi/envs/default/bin/python" \
  "$BASE/waymo23-native-import/gpu_test_service.py" "$@"
