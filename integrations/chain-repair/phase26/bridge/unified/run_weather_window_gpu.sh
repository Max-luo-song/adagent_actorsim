#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# New 3x3 framework-render window only; no full sequence, training or UniAD.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
unset PYTHONPATH PHASE26_PYTHON
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$(hugsim_source_path "$BRIDGE/env_fix/weather_mpl_cache")"
bash "$(hugsim_source_path "$BRIDGE/env_fix/setup_weather_env.sh")"
"$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" -B -c 'import cv2; print("Optical-flow dependency:",cv2.__version__)'
"$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" -B -m unittest discover -s "$(hugsim_source_path "$BRIDGE/weather_sequence")" -p 'test_*.py' -v
OUT="${1:-$HERE/weather_window_runs/$(date -u +%Y%m%dT%H%M%SZ)_rainy_3x3}"
printf 'Weather window output: %s\n' "$OUT"
exec bash "$(hugsim_source_path "$HERE/run_phase26.sh")" weather-window "$OUT"
