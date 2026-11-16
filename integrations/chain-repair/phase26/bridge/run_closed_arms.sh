#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# Phase26 stage-1 diagnostic closed-loop arms A/B/C (UNCERTIFIED; N1 suspended).
# Reuses the reviewed isolated phase11 paired-arm harness (soft recorded road
# constraint; collision/actor-overlap guards intact) — the same harness that
# produced the existing 552 baseline. The repo-level probe currently hard-stops
# scene-552 frame-80 runs at step 0 even for the known-good old asset
# (fixed_initial_drivable_conflict), so the allowlisted service dispatch cannot
# produce usable 552 closed-loop data; see HANDOFF.md.
#
# Gate: run ONLY after the three smoke jobs passed:
#   srveh552-smoke-20261027-v1, srveh552-smoke80-20261027-v1,
#   srveh552-smoke140overlay-20261027-v1
# Usage (GPU terminal):
#   bash run_closed_arms.sh [A|B|C|all]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HUGSIM_ROOT}"
P11="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase11")"
PY="${HUGSIM_PYTHON}"
SEL="${1:-all}"

# wait until the gpu_test_service worker releases its lock (avoid GPU contention)
while ! "$PY" -c "import fcntl,sys; f=open('$ROOT/outputs/gpu_test_service/worker.lock'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; do
  echo "GPU service worker still active; waiting 30s (or stop it: touch $ROOT/outputs/gpu_test_service/STOP)"
  sleep 30
done

# smoke gate
S="$(hugsim_source_path "$ROOT/outputs/gpu_test_service/jobs")"
for j in srveh552-smoke-20261027-v1 srveh552-smoke80-20261027-v1 srveh552-smoke140overlay-20261027-v1; do
  st="$(hugsim_source_path "$S/$j/status.json")"
  if [[ ! -f "$st" ]] || ! grep -q '"status": "passed"' "$st"; then
    echo "GATE: smoke $j not passed; refusing to start closed loops" >&2; exit 1
  fi
done

run_arm() {
  local arm="$1"
  local out="$(hugsim_source_path "$HERE/closed_arms/arm_${arm}")"
  if [[ -e "$out" ]]; then echo "arm $arm output exists; refusing to overwrite: $out" >&2; exit 1
  fi
  echo "=== arm $arm -> $out"
  "$PY" -u -B "$(hugsim_source_path "$P11/run_arm.py")" "$(hugsim_source_path "$HERE/requests/closed552_arm_${arm}.json")" "$out"
}
case "$SEL" in
  A|B|C) run_arm "$SEL" ;;
  all) run_arm A; run_arm B; run_arm C ;;
  *) echo "usage: $0 [A|B|C|all]" >&2; exit 1 ;;
esac
