#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Copyright (c) Facebook, Inc. and its affiliates.
set -ex

ldconfig  # https://github.com/NVIDIA/nvidia-docker/issues/854

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$(hugsim_source_path "$script_dir/pkg_helpers.bash")"

echo "Build Settings:"
echo "CU_VERSION: $CU_VERSION"                 # e.g. cu101
echo "D2_VERSION_SUFFIX: $D2_VERSION_SUFFIX"   # e.g. +cu101 or ""
echo "PYTHON_VERSION: $PYTHON_VERSION"         # e.g. 3.7
echo "PYTORCH_VERSION: $PYTORCH_VERSION"       # e.g. 1.4

setup_cuda
setup_wheel_python

yum install ninja-build -y
ln -sv /usr/bin/ninja-build /usr/bin/ninja || true

pip_install pip numpy -U
pip_install "torch==$PYTORCH_VERSION" \
  -f https://download.pytorch.org/whl/"$CU_VERSION"/torch_stable.html

# use separate directories to allow parallel build
BASE_BUILD_DIR=build/$CU_VERSION-py$PYTHON_VERSION-pt$PYTORCH_VERSION
python setup.py \
  build -b "$BASE_BUILD_DIR" \
  bdist_wheel -b "$(hugsim_source_path "$BASE_BUILD_DIR/build_dist")" -d "wheels/$CU_VERSION/torch$PYTORCH_VERSION"
rm -rf "$BASE_BUILD_DIR"
