#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# 框架后端闭环联通探针（诊断、未认证）：HUGSIM 状态推进 + 自研渲染观测 +
# 固定 UniAD + iLQR 跟踪直通（无避碰、不与 MPC 等价）。unedited 基线臂 5 步。
# 验收核心：每步观测由执行后的自车位姿重新渲染（证据：图像随位姿变化 +
# 包内 camera_to_world 与 motion 一致）；禁止录制位姿/预渲染替代。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
ROOT="${HUGSIM_ROOT}"
P11="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase11")"
PY="${HUGSIM_PYTHON}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$BRIDGE/framework_backend:$BRIDGE/../framework_repo:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"

if pgrep -f "run_arm.py|run_arm_opt.py|run_arm_tracking.py|run_arm_posetrack.py|run_arm_fwbackend.py|paired_closed_loop" >/dev/null 2>&1; then
  echo "检测到正在运行的闭环进程；等其结束后重试" >&2; exit 1
fi
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker active; waiting 30s"; sleep 30
done

OUT="$(hugsim_source_path "$HERE/closed_arms/fastprobe_fwbackend_v1")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

"$PY" - <<PYEOF
import hashlib, json
from pathlib import Path
files = ["$(hugsim_source_path "$HERE/run_arm_fwbackend.py")", "$(hugsim_source_path "$HERE/scene_api_fwbackend.py")",
         "$(hugsim_source_path "$HERE/backend.py")",
         "$(hugsim_source_path "$BRIDGE/fast_control/tracking_common.py")",
         "$(hugsim_source_path "$BRIDGE/fast_control/planning_guard_tracking.py")",
         "$(hugsim_source_path "$P11/paired_closed_loop_arm.py")",
         "$(hugsim_source_path "$ROOT/integrations/sr-hugsim/scene_api.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/models/gaussians/vanilla.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/models/trainers/base.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/models/trainers/scene_graph.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/models/modules.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/datasets/base/pixel_source.py")",
         "$(hugsim_source_path "$BRIDGE/../framework_repo/datasets/waymo/waymo_sourceloader.py")"]
missing = [f for f in files if not Path(f).exists()]
if missing: raise SystemExit('missing: %s' % missing)
fp = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
fp.update(threads=dict(OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1),
          observation_backend='framework renderer (bridge/framework_backend)',
          control_path='tracking_passthrough_no_obstacle_optimization_v1',
          safety_equivalence_to_mpc=False,
          acceptance='per-step observation re-rendered from executed ego pose; '
                     'no recorded future poses; no prerendered video',
          formal_evaluation_admitted=False, n1_certified=False)
Path("$(hugsim_source_path "$OUT/perf_fingerprint.json")").write_text(json.dumps(fp, indent=2))
print('fingerprint written')
PYEOF

"$PY" -u -B "$(hugsim_source_path "$HERE/run_arm_fwbackend.py")" "$(hugsim_source_path "$HERE/requests/fastprobe_fwbackend.json")" "$(hugsim_source_path "$OUT/arm")"
