#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# nvdiffrast 隔离构建与功能冒烟（GPU 终端；不修改 HUGSIM/UniAD 环境）。
# v3（2026-10-30）：0.4.0 的 C 扩展 _nvdiffrast_c 需安装期编译（setup.py
# CUDAExtension）。在隔离目录复制源码并构建，不动 vendor 源树。
# 版本偏差如实记录：框架 pin diffrp-nvdiffrast==0.3.3.1（fork，离线不可得）；
# 本脚本用本地官方 nvdiffrast 0.4.0 源码——诊断兼容性候选，非等价替代。
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
NVDIFFRAST_SRC=${HUGSIM_EXTERNAL_ROOT}/adscenegraph/tempgls/BezierGS/nvdiffrast
BUILD_SRC="$(hugsim_source_path "$HERE/nvdiffrast_build_src")"

# CUDA 12.4 工具链对齐 torch cu124（系统默认 nvcc 11.1 会被拒）
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$HERE/torch_ext_isolated")"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# ① 隔离复制源码 + 构建（首次约几分钟；已建则复用）
if [ ! -d "$BUILD_SRC" ]; then
  cp -r "$NVDIFFRAST_SRC" "$BUILD_SRC"
fi
cd "$BUILD_SRC"
if ! ls build/lib.linux-*/_nvdiffrast_c*.so >/dev/null 2>&1; then
  "$(hugsim_source_path "$CUDA_HOME/bin/nvcc")" --version | tail -1
  "$PY" setup.py build_ext 2>&1 | tail -20
fi
LIBDIR=$(ls -d "$BUILD_SRC"/build/lib.linux-*)
echo "built libdir: $LIBDIR"

# ② 隔离元数据（dist-info，内容与实际源码一致）
META_DIR="$(hugsim_source_path "$HERE/nvdiffrast_meta/nvdiffrast-0.4.0.dist-info")"
mkdir -p "$META_DIR"
cat > "$(hugsim_source_path "$META_DIR/METADATA")" << 'META'
Metadata-Version: 2.1
Name: nvdiffrast
Version: 0.4.0
Summary: Official nvdiffrast 0.4.0 source build (BezierGS vendored copy); metadata generated locally. Diagnostic compatibility candidate; NOT the pinned diffrp-nvdiffrast==0.3.3.1 fork.
META
echo "nvdiffrast" > "$(hugsim_source_path "$META_DIR/top_level.txt")"

# ③ 功能冒烟（首次触发 torch 插件 JIT，编译进隔离缓存）
export PYTHONPATH="$BUILD_SRC:$LIBDIR:$HERE/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"$PY" "$(hugsim_source_path "$HERE/test_texture.py")"
