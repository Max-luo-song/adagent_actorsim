#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# pose_track + 跟踪直通 5 步闭环探针（诊断模式、未经认证）。
# 运行中更新（轨迹推进，无 revision 变化）+ 轻量控制（iLQR 跟踪，
# 无避碰优化、不与 MPC 安全等价，tracking_passthrough_unverified）。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
ROOT="${HUGSIM_ROOT}"
P11="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase11")"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if pgrep -f "run_arm.py|run_arm_opt.py|run_arm_tracking.py|run_arm_posetrack.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；等其结束后重试" >&2; exit 1
fi
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done

OUT="$(hugsim_source_path "$HERE/closed_arms/fastprobe_A5_posetrack_v1")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

"$PY" - <<PYEOF
import hashlib, json
from pathlib import Path
files = ["$(hugsim_source_path "$HERE/run_arm_posetrack.py")", "$(hugsim_source_path "$HERE/scene_api_posetrack.py")",
         "$(hugsim_source_path "$BRIDGE/fast_control/tracking_common.py")",
         "$(hugsim_source_path "$BRIDGE/fast_control/planning_guard_tracking.py")",
         "$(hugsim_source_path "$BRIDGE/pose_track/pose_track.py")", "$(hugsim_source_path "$BRIDGE/pose_track/session_adapter.py")",
         "$(hugsim_source_path "$BRIDGE/pose_track/gpu_applier.py")",
         "$(hugsim_source_path "$HERE/requests/closed552_insert_realcar_track.json")",
         "$(hugsim_source_path "$P11/paired_closed_loop_arm.py")", "$(hugsim_source_path "$P11/closed172_common.py")",
         "$(hugsim_source_path "$P11/drivable_area.py")", "$(hugsim_source_path "$P11/road_route_seed.py")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/planning_guard.py")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/trajectory_control.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/scene_api.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/native_scene_backend.py")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/gs.pth")",
         "$(hugsim_source_path "$ROOT/data/3DRealCar/2024_06_04_13_44_39/wlh.json")",
         "$(hugsim_source_path "$ROOT/waymo23-native-import/configs/asset_calibrations/2024_06_04_13_44_39-support-v1.json")"]
missing = [f for f in files if not Path(f).exists()]
if missing: raise SystemExit('missing: %s' % missing)
fp = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
fp.update(threads=dict(OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1),
          control_path='tracking_passthrough_no_obstacle_optimization_v1',
          pose_track='advance-on-render via scene_api_posetrack; strict window [8.0,8.6]s; anchor-local +x 2 m/s',
          safety_equivalence_to_mpc=False,
          scope='5-step closed loop: moving managed asset + tracking control; NOT an avoidance proof',
          formal_evaluation_admitted=False, n1_certified=False)
Path("$(hugsim_source_path "$OUT/perf_fingerprint.json")").write_text(json.dumps(fp, indent=2))
print('fingerprint written')
PYEOF

"$PY" -u -B "$(hugsim_source_path "$HERE/run_arm_posetrack.py")" "$(hugsim_source_path "$HERE/requests/fastprobe_A5_posetrack.json")" "$(hugsim_source_path "$OUT/arm")"
