#!/usr/bin/env bash
# Execute in the GPU terminal. Uses existing environments; installs nothing.
set -euo pipefail
MODE="${1:-all}"
if [[ $# -gt 1 || ( "$MODE" != "all" && "$MODE" != "--uniad-only" ) ]]; then
    echo "Usage: bash waymo23-native-import/run_gpu_smoke.sh [--uniad-only]" >&2
    exit 2
fi
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
JOB="$(mktemp -d "$BASE/outputs/gpu_smoke_$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"
export MPLCONFIGDIR="$JOB/matplotlib"
printf '%s\n' "$JOB" > "$BASE/outputs/latest_gpu_smoke.txt"
exec > >(tee "$JOB/run.log") 2>&1
echo "OUTPUT: $JOB"
trap 'rc=$?; printf "%s\n" "$rc" > "$JOB/exit_code.txt"; echo "Exit: $rc; logs: $JOB"' EXIT
HUGPY="$BASE/code/HUGSIM/.pixi/envs/default/bin/python"
UNIPY="$BASE/tools/pixi-uniad-legacy/.pixi/envs/default/bin/python"
cd "$BASE"
nvidia-smi
PYTHONS=("$UNIPY")
if [[ "$MODE" == "all" ]]; then PYTHONS=("$HUGPY" "$UNIPY"); fi
for py in "${PYTHONS[@]}"; do
    env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME "$py" -c \
      'import torch; print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"devices",torch.cuda.device_count(),flush=True); assert torch.cuda.is_available(), "CUDA unavailable in this environment"; x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name(),x.item(),flush=True)'
done
if [[ "$MODE" == "all" ]]; then
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME "$HUGPY" \
    -m unittest discover -s waymo23-native-import/tests -p 'test_*.py' -v
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME "$HUGPY" \
    "$BASE/waymo23-native-import/gpu_render_smoke.py" \
    --source-report "$BASE/outputs/waymo552_three_setup_20261018_043131/source_report.json" \
    --output "$JOB/render"
fi
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME "$UNIPY" \
    "$BASE/waymo23-native-import/run_reliable_sequence.py" \
    --sequence-report "$BASE/outputs/waymo552_baseline_sequence_20261018_043758/sequence_report.json" \
    --max-frames 3 --output "$JOB/uniad"
echo "GPU smoke tests completed. N1 real-edit certification and feedback closed loop have not been run."
