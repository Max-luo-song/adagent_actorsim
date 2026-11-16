#!/bin/bash -e

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Copyright (c) Facebook, Inc. and its affiliates.

# Download the mini dataset (coco val2017_100, with only 100 images)
# to be used in unittests & integration tests.

cd "${0%/*}"

BASE=https://dl.fbaipublicfiles.com/detectron2
ROOT="${HUGSIM_ROOT}"
ROOT="${HUGSIM_ROOT}"
mkdir -p $ROOT/coco/annotations

for anno in instances_val2017_100 \
  person_keypoints_val2017_100 ; do

  dest=$ROOT/coco/annotations/$anno.json
  [[ -s $dest ]] && {
    echo "$dest exists. Skipping ..."
  } || {
    wget $BASE/annotations/coco/$anno.json -O $dest
  }
done

dest=$ROOT/coco/val2017_100.tgz
[[ -d $ROOT/coco/val2017 ]] && {
  echo "$(hugsim_source_path "$ROOT/coco/val2017 exists. Skipping ...")"
} || {
  wget $BASE/annotations/coco/val2017_100.tgz -O $dest
  tar xzf $dest -C $ROOT/coco/ && rm -f $dest
}
