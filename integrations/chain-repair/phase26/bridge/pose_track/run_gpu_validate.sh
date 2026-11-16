#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# pose_track 短时 GPU 验证（纯渲染+位姿连续性；不接 UniAD、不做控制）。
# 诊断模式、未经认证。不覆盖已有输出（gpu_evidence/ 存在即拒绝）。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if pgrep -f "run_arm.py|run_arm_opt.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；等其结束后重试" >&2; exit 1
fi
while ! "$PY" -c "import fcntl; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done
exec "$PY" -u -B "$(hugsim_source_path "$HERE/gpu_validate.py")"
