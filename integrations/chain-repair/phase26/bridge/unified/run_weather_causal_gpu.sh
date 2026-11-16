#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Reuse measured 3x3 renders; only new causal weather inference is scheduled.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
unset PYTHONPATH PHASE26_PYTHON
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$(hugsim_source_path "$BRIDGE/env_fix/weather_mpl_cache")"
bash "$(hugsim_source_path "$BRIDGE/env_fix/setup_weather_env.sh")"
"$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" -B -c 'import cv2; print("Optical flow:",cv2.__version__)'
"$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" -B -m unittest discover -s "$(hugsim_source_path "$BRIDGE/weather_sequence")" -p 'test_*.py' -v
INPUTS="$(hugsim_source_path "$HERE/weather_window_runs/20261101T112744Z_rainy_3x3/inputs")"
OUT="${1:-$HERE/weather_causal_runs/$(date -u +%Y%m%dT%H%M%SZ)_rainy}"
printf 'Causal weather output: %s\n' "$OUT"
exec bash "$(hugsim_source_path "$HERE/run_phase26.sh")" weather-causal "$INPUTS" "$OUT"
