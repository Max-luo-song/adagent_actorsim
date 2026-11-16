#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export LD_LIBRARY_PATH="${ROOT}/libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QCRAFT_RELEASE=1
export Q_BIN_PREFIX="${ROOT}"
# sensor_scenario_config.pb.txt is loaded via relative path
# onboard/lite/scenario_config/...
cd "${ROOT}"
exec "${ROOT}/dump_camera_image_virtual" "$@"
