#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
LOCAL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:-audit}" in
 audit) exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/audit_recordings.py")";;
 replay) exec "$PY" -u -B "$(hugsim_source_path "$LOCAL/replay_recordings.py")" "${2:?case required}";;
 *) echo 'Offline candidate only: audit | replay CASE. Fresh GPU validation not yet admitted.' >&2; exit 2;;
esac
