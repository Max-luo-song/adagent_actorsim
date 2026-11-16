#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
ROOT="${HUGSIM_ROOT}"
export LD_LIBRARY_PATH="${ROOT}/libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QCRAFT_RELEASE=1
export Q_BIN_PREFIX="${ROOT}"
# sensor_scenario_config.pb.txt is loaded via relative path
# onboard/lite/scenario_config/...
cd "${ROOT}"
exec "$(hugsim_source_path "${ROOT}/dump_camera_image_virtual")" "$@"
