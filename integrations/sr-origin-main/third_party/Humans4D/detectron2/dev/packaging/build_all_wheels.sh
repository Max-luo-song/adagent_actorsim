#!/bin/bash -e

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Copyright (c) Facebook, Inc. and its affiliates.

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at detectron2 root!"
  exit 1
}

build_one() {
  cu=$1
  pytorch_ver=$2

  case "$cu" in
    cu*)
      container_name=manylinux-cuda${cu/cu/}
      ;;
    cpu)
      container_name=manylinux-cuda101
      ;;
    *)
      echo "Unrecognized cu=$cu"
      exit 1
      ;;
  esac

  echo "Launching container $container_name ..."
  container_id="$container_name"_"$cu"_"$pytorch_ver"

  py_versions=(3.7 3.8 3.9)

  for py in "${py_versions[@]}"; do
    docker run -itd \
      --name "$container_id" \
      --mount type=bind,source="$(pwd)",target=/detectron2 \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_id sh
      export CU_VERSION=$cu D2_VERSION_SUFFIX=+$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      cd /detectron2 && ./dev/packaging/build_wheel.sh
EOF

    docker container stop $container_id
    docker container rm $container_id
  done
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  build_one cu113 1.10
  build_one cu111 1.10
  build_one cu102 1.10
  build_one cpu 1.10

  build_one cu111 1.9
  build_one cu102 1.9
  build_one cpu 1.9

  build_one cu111 1.8
  build_one cu102 1.8
  build_one cu101 1.8
  build_one cpu 1.8
fi
