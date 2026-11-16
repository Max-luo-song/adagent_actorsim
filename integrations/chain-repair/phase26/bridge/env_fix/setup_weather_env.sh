#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Isolated Python packages; existing CUDA torch is inherited read-only.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
BASE_PY="$(hugsim_source_path "$ROOT/code/HUGSIM/.pixi/envs/default/bin/python")"
VENV="$(hugsim_source_path "$HERE/weather_env_v1")"
exec 8>"$(hugsim_source_path "$HERE/weather_setup.lock")"
flock -n -x 8 || { echo 'Weather environment setup already running' >&2; exit 1; }
if [[ ! -e "$VENV" ]]; then
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
PY="${HUGSIM_PYTHON}"
"$BASE_PY" - <<'PYEOF' > "$(hugsim_source_path "$HERE/weather_constraints.txt")"
import importlib.metadata as md
for name in ('torch','torchvision','numpy'):
    print(f'{name}=={md.version(name)}')
PYEOF
# If already installed and import-tested, reuse without network/pip activity.
if ! "$PY" -B "$(hugsim_source_path "$HERE/verify_weather_env.py")" >/dev/null 2>&1; then
  PIP_SOURCE=()
  if [[ -n "${PHASE26_WHEELHOUSE:-}" ]]; then
    PIP_SOURCE=(--no-index --find-links "$PHASE26_WHEELHOUSE")
  fi
  "$PY" -m pip install --disable-pip-version-check "${PIP_SOURCE[@]}" \
    -c "$(hugsim_source_path "$HERE/weather_constraints.txt")" -r "$(hugsim_source_path "$HERE/weather_requirements.txt")"
fi
"$PY" -B "$(hugsim_source_path "$HERE/verify_weather_env.py")" | tee "$(hugsim_source_path "$HERE/weather_env_verified.json")"
