#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot

source ./docker_base.sh

function stop_docker() {
  echo "stop container in workspace: ${WS_DIR}"
  local docker_info=$(fetch_docker_info)
  local container_id=$(echo "${docker_info}" | awk -F'|' '{print $1}')
  if [ "${container_id}" != "" ]; then
    ${DOCKER_CMD} stop ${container_id}
  fi
}

stop_docker
