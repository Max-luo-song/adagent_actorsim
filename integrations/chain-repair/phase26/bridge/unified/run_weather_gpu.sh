#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Setup small dependencies, verify imports, then run one locked offline image.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$(hugsim_source_path "$BRIDGE/env_fix/weather_mpl_cache")"
export PYTHONDONTWRITEBYTECODE=1
# Avoid inherited package overlays; this job has its own venv.
unset PYTHONPATH
bash "$(hugsim_source_path "$BRIDGE/env_fix/setup_weather_env.sh")"
export PHASE26_PYTHON="$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")"
OUT="${1:-$HERE/weather_runs/$(date -u +%Y%m%dT%H%M%SZ)_rainy}"
printf 'Weather output: %s\n' "$OUT"
exec bash "$(hugsim_source_path "$HERE/run_phase26.sh")" weather --weather rainy --output "$OUT"
