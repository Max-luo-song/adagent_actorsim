#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot

# Usage: ./retry.sh <command>

set +e

max=3
interval=5

n=0
until "$@"; do
  n=$((n+1))
  if [ $n -ge $max ]; then
    echo "Command failed after $max attempts."
    exit 1
  fi
  echo "Command failed. Retry #$n in $interval seconds..."
  sleep $interval
done
