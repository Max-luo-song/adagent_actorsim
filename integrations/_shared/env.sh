#!/usr/bin/env bash
# Source-only environment boundary. No network, GPU work or task startup.
_hugsim_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HUGSIM_ROOT="${HUGSIM_ROOT:-$(cd -- "$_hugsim_env_dir/../.." && pwd)}"
export HUGSIM_OUTPUT_ROOT="${HUGSIM_OUTPUT_ROOT:-$HUGSIM_ROOT/outputs}"
export HUGSIM_EXTERNAL_ROOT="${HUGSIM_EXTERNAL_ROOT:-$HUGSIM_ROOT/assets/external}"
export HUGSIM_CHAIN_CODE_ROOT="${HUGSIM_CHAIN_CODE_ROOT:-$HUGSIM_ROOT/integrations/chain-repair}"
export PYTHONPATH="$HUGSIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${HUGSIM_PYTHON:-}" ]]; then
  if [[ -x "$HUGSIM_ROOT/code/HUGSIM/.pixi/envs/default/bin/python" ]]; then
    export HUGSIM_PYTHON="$HUGSIM_ROOT/code/HUGSIM/.pixi/envs/default/bin/python"
  else
    export HUGSIM_PYTHON="$(command -v python3)"
  fi
fi
hugsim_source_path() {
  local requested="$1" prefix="$HUGSIM_OUTPUT_ROOT/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair"
  if [[ "$requested" == "$prefix"* ]]; then
    local published="$HUGSIM_CHAIN_CODE_ROOT${requested#"$prefix"}"
    if [[ -e "$published" ]]; then printf '%s\n' "$published"; return; fi
  fi
  if [[ "$requested" == "$HUGSIM_CHAIN_CODE_ROOT"* && ! -e "$requested" ]]; then
    printf '%s\n' "$prefix${requested#"$HUGSIM_CHAIN_CODE_ROOT"}"; return
  fi
  printf '%s\n' "$requested"
}
hugsim_pythonpath() {
  local entry mapped result=""
  local -a entries
  IFS=: read -r -a entries <<< "$1"
  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    mapped="$(hugsim_source_path "$entry")"
    result="${result:+$result:}$mapped"
    # Compiled extensions are external build artifacts, not copied sources.
    if [[ "$mapped" != "$entry" ]]; then result="$result:$entry"; fi
  done
  printf '%s\n' "$result"
}
unset _hugsim_env_dir
