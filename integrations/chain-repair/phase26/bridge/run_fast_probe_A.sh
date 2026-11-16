#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 5 步短时工程验证（渲染→感知→动作→状态反馈）。不证明绕行成功。
# 资产：3DRealCar 2024_06_04_13_44_39（360°物体重建，已通过目标视角渲染验收；
# 不使用 id25 导出资产的新视角退化画面作为主线验收）。
# 性能版本：已验证优化版 clearance/记忆化（bridge/opt/）+ BLAS=1。
# 与既有结果完全隔离：独立输出目录，不覆盖、不混用、不重复提交。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
P11="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase11")"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 1) GPU 竞争保护：本机已有闭环/harness 进程在跑时拒绝启动（避免污染双方计时）
if pgrep -f "run_arm.py|run_arm_opt.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；为隔离计时与显存，等其结束后重试" >&2; exit 1
fi
# 2) 服务 worker 锁等待
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done

OUT="$(hugsim_source_path "$HERE/closed_arms/fastprobe_A5_opt_v1")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

# 3) 执行前指纹：实际加载的模块文件 + 资产 + 线程配置
"$PY" - <<PYEOF
import hashlib, json
from pathlib import Path
files = ["$(hugsim_source_path "$HERE/opt/run_arm_opt.py")", "$(hugsim_source_path "$HERE/opt/closed172_common_opt.py")",
         "$(hugsim_source_path "$HERE/opt/temporal_occupancy_fast.py")",
         "$(hugsim_source_path "$P11/paired_closed_loop_arm.py")", "$(hugsim_source_path "$P11/run_arm.py")",
         "$(hugsim_source_path "$P11/drivable_area.py")", "$(hugsim_source_path "$P11/road_route_seed.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/native_scene_backend.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/scene_api.py")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/gs.pth")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/wlh.json")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json")"]
fp = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
fp.update(threads=dict(OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1),
          performance_version='opt_clearance_memoized_v1_blas1_fastprobe',
          asset='3DRealCar 2024_06_04_13_44_39 (360 object reconstruction; passed target-view render acceptance)',
          scope='5-step engineering probe; NOT an avoidance proof; isolated from arms A/B/C',
          formal_evaluation_admitted=False, n1_certified=False)
Path("$(hugsim_source_path "$OUT/perf_fingerprint.json")").write_text(json.dumps(fp, indent=2))
print(json.dumps({k: v[:12] for k, v in list(fp.items())[:4]}, indent=1))
PYEOF

"$PY" -u -B "$(hugsim_source_path "$HERE/opt/run_arm_opt.py")" "$(hugsim_source_path "$HERE/requests/fastprobe_A5.json")" "$(hugsim_source_path "$OUT/arm")"
