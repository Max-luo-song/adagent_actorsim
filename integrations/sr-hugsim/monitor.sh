#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE"
while true; do
    date -u '+%Y-%m-%d %H:%M:%S UTC'
    if [[ -f outputs/gpu_test_service/heartbeat.json ]]; then
        cat outputs/gpu_test_service/heartbeat.json
    fi
    for scene in 172 552; do
        identity="srpbr-${scene}-20261021-v1"
        status="outputs/gpu_test_service/jobs/${identity}/status.json"
        printf '\nScene %s: ' "$scene"
        if [[ -f "$status" ]]; then
            cat "$status"
        elif [[ -f "outputs/gpu_test_service/requests/${identity}.json" ]]; then
            printf 'queued\n'
        else
            printf 'no request or status found\n'
        fi
    done
    printf '\n'
    sleep 5
done
