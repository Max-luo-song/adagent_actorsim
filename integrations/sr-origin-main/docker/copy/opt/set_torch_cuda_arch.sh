#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot

arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u 2>/dev/null | tr '\n' ';' | sed 's/;$//')

if [ -z "$arch" ]; then
    arch=""
fi

if ! grep -q "TORCH_CUDA_ARCH_LIST" /etc/profile; then
    echo "export TORCH_CUDA_ARCH_LIST=\"$arch\"" >> /etc/profile
    echo "TORCH_CUDA_ARCH_LIST set to $arch in /etc/profile"
else
    echo "TORCH_CUDA_ARCH_LIST already set in /etc/profile"
fi
