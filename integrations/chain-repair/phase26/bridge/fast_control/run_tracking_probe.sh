#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 跟踪直通（无 MPC）5 步工程探针：渲染→UniAD(固定)→iLQR 跟踪→真实动作执行→
# 状态反馈→碰撞记录→终止机制。诊断模式、未经认证。
# 明确声明：本配置不提供 evidence-tiered MPC 的避碰优化能力，不与 MPC 安全等价；
# 每步碰撞记录为后验审计，不是避碰认证。run_report trace 的 mode 字段中
# " / MPC" 后缀为封存 harness 的固定字符串，本臂未运行 MPC——以各步
# control_decision.json 的 policy/control_path 字段为准。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
P11="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase11")"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if pgrep -f "run_arm.py|run_arm_opt.py|run_arm_tracking.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；为隔离计时与显存，等其结束后重试" >&2; exit 1
fi
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done

OUT="$(hugsim_source_path "$HERE/closed_arms/fastprobe_A5_tracking_v1")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

"$PY" - <<PYEOF
import hashlib, json
from pathlib import Path
files = ["$(hugsim_source_path "$HERE/run_arm_tracking.py")", "$(hugsim_source_path "$HERE/tracking_common.py")",
         "$(hugsim_source_path "$HERE/planning_guard_tracking.py")",
         "$(hugsim_source_path "$P11/paired_closed_loop_arm.py")", "$(hugsim_source_path "$P11/closed172_common.py")",
         "$(hugsim_source_path "$P11/drivable_area.py")", "$(hugsim_source_path "$P11/road_route_seed.py")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/planning_guard.py")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/trajectory_control.py")",
         "$(hugsim_source_path "$ROOT/code/HUGSIM/sim/ilqr/lqr.py")",
         "$(hugsim_source_path "$ROOT/code/HUGSIM/sim/ilqr/lqr_solver.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/native_scene_backend.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/scene_api.py")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/gs.pth")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/wlh.json")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json")"]
fp = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
fp.update(threads=dict(OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1),
          control_path='tracking_passthrough_no_obstacle_optimization_v1',
          safety_equivalence_to_mpc=False,
          scope='5-step engineering probe of the tracking control path; NOT an avoidance proof; '
                'no obstacle optimization; collision records are post-hoc audit only',
          formal_evaluation_admitted=False, n1_certified=False)
Path("$(hugsim_source_path "$OUT/perf_fingerprint.json")").write_text(json.dumps(fp, indent=2))
print(json.dumps({k: v[:12] for k, v in list(fp.items())[:4]}, indent=1))
PYEOF

"$PY" -u -B "$(hugsim_source_path "$HERE/run_arm_tracking.py")" "$(hugsim_source_path "$HERE/requests/fastprobe_A5_tracking.json")" "$(hugsim_source_path "$OUT/arm")"
