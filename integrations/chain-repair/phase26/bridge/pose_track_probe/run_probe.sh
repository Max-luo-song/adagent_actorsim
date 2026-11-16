#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# pose_track 动态轨迹接入 5-8 观测渲染探针。纯渲染循环，无 UniAD 推理、无 MPC。
# 不验证绕行、不验证模型历史连续性、不出 N1 证书。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 1) GPU 竞争保护
if pgrep -f "run_arm.py|run_arm_opt.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；等其结束后重试" >&2; exit 1
fi
# 2) 服务 worker 锁等待
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done

OUT="$(hugsim_source_path "$HERE/pose_track_probe_result")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi

# 3) 执行前指纹（探针代码 + 轨迹库 + 资产）
"$PY" - <<PYEOF
import hashlib, json
from pathlib import Path
files = ["$(hugsim_source_path "$HERE/pose_track_probe.py")", "$(hugsim_source_path "$HERE/plan.json")",
         "$(hugsim_source_path "$HERE/../pose_track/pose_track.py")", "$(hugsim_source_path "$HERE/../pose_track/session_adapter.py")",
         "$(hugsim_source_path "$HERE/../pose_track/gpu_applier.py")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/gs.pth")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json")"]
fp = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
fp.update(threads=dict(OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1),
          scope='pose track hookup probe; render loop only; NOT UniAD history validation',
          formal_evaluation_admitted=False, n1_certified=False)
Path("$(hugsim_source_path "$HERE/pose_track_probe_fingerprint.json")").write_text(json.dumps(fp, indent=2))
print("fingerprint written")
PYEOF

"$PY" -u -B "$(hugsim_source_path "$HERE/pose_track_probe.py")" --plan "$(hugsim_source_path "$HERE/plan.json")" --output "$OUT"
