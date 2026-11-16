#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
RESUME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="$(hugsim_source_path "$RESUME_DIR/../runs/cutin551_white_edge_resumable_v1")"
mkdir -p "$STATE_ROOT"
nohup bash "$(hugsim_source_path "$RESUME_DIR/run.sh")" </dev/null >>"$(hugsim_source_path "$STATE_ROOT/console.log")" 2>&1 &
RESUME_PID=$!
printf '%s\n' "$RESUME_PID" >"$(hugsim_source_path "$STATE_ROOT/launcher.pid")"
printf 'Background launcher PID: %s\nLog: %s\nRepeat this same command after a stopped GPU job to resume.\n' "$RESUME_PID" "$(hugsim_source_path "$STATE_ROOT/console.log")"
