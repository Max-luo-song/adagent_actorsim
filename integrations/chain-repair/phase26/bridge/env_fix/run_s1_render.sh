#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# s1 全场景 PBR 短渲染（全序列 198 帧 × 3 相机；诊断模式、未经认证）。
# 依赖：nvdiffrast 隔离构建（setup_nvdiffrast.sh 已成功）+ GPU 节点 libGL。
# 无编辑（edit_none.yaml 无 Nodes）；不修改 HUGSIM/UniAD 环境。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
ROOT="${HUGSIM_ROOT}"
F="$(hugsim_source_path "$BRIDGE/../framework_repo")"
PY="${HUGSIM_PYTHON}"
CKPT=${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$HERE/torch_ext_isolated")"
LIBDIR=$(ls -d "$HERE"/nvdiffrast_build_src/build/lib.linux-*)
BUILD_SRC="$(hugsim_source_path "$HERE/nvdiffrast_build_src")"
export PYTHONPATH="$F:$BUILD_SRC:$LIBDIR:$HERE/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# SMPL 模板：官方 pkl 需 chumpy（不可得），用已验证的受控转换 npz
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"

OUT="$(hugsim_source_path "$BRIDGE/demo/s1_fullscene_v1")"
if [[ -e "$OUT" ]]; then echo "refusing to overwrite: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"

# 前置：导入链 + 数据目录（2026-10-31 起用训练仓原生 552 数据：含 humanpose/smpl.pkl，
# 存档 inputs 副本缺该文件）
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "$F")
import nvdiffrast.torch  # noqa
import datasets.driving_dataset  # noqa
import models.trainers  # noqa
print("preflight imports OK")
from pathlib import Path
for p in ("$(hugsim_source_path "$BRIDGE/framework_data/552/images")", "$(hugsim_source_path "$BRIDGE/framework_data/552/ego_pose")",
          "$(hugsim_source_path "$BRIDGE/framework_data/552/instances")"):
    assert Path(p).is_dir(), p
print("data dirs OK")
PYEOF

cd "$F"   # render_edit.py 以相对路径读 ./configs/particle_config.yaml
"$PY" sim_render/cam/render_edit.py \
  --resume_from "$CKPT" \
  --edit_config "$(hugsim_source_path "$BRIDGE/framework_data/edit_none.yaml")" \
  --render_cam_ids 0 1 2 \
  --post_fix _s1_baseline \
  log_dir="$OUT" \
  data.data_root=${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/processed/training \
  render.render_full=true render.render_test=false
echo "s1 render done -> $OUT"
