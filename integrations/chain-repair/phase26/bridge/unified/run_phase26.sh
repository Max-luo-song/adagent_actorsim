#!/usr/bin/env bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
# One entry: GPU articulation acceptance -> edited short closed loop.
# Existing full renders and successful baseline probes are not rerun.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$(dirname "$HERE")"
ROOT="${HUGSIM_ROOT}"
PY="${HUGSIM_PYTHON}"
MODE="${1:-status}"
if [[ "$MODE" == scene-task-stdio ]]; then MODE=agent-json-stdio; fi
if [[ "$MODE" == scene-task || "$MODE" == scene-task-check ]]; then
  if [[ "$MODE" == scene-task ]]; then MODE=agent-json; else MODE=agent-json-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/agent_json/examples/unified_scene_task.json")"; fi
fi
if [[ "$MODE" == search-loop || "$MODE" == search-loop-check ]]; then
  if [[ "$MODE" == search-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/source_insert551_motion_search_v3.json")"; fi
fi
if [[ "$MODE" == search-diagnostic || "$MODE" == search-diagnostic-check ]]; then
  if [[ "$MODE" == search-diagnostic ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/diagnostic551_motion_search_v3_10s.json")"; fi
fi
if [[ "$MODE" == envelope-loop || "$MODE" == envelope-loop-check ]]; then
  if [[ "$MODE" == envelope-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/source_insert551_motion_envelope_v2.json")"; fi
fi
if [[ "$MODE" == envelope-diagnostic || "$MODE" == envelope-diagnostic-check ]]; then
  if [[ "$MODE" == envelope-diagnostic ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/diagnostic551_motion_envelope_v2_10s.json")"; fi
fi
if [[ "$MODE" == memory-loop || "$MODE" == memory-loop-check ]]; then
  if [[ "$MODE" == memory-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/source_insert551_memory_lifecycle.json")"; fi
fi
if [[ "$MODE" == memory-diagnostic || "$MODE" == memory-diagnostic-check ]]; then
  if [[ "$MODE" == memory-diagnostic ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/diagnostic551_memory_lifecycle_10s.json")"; fi
fi
if [[ "$MODE" == road-runtime-loop || "$MODE" == road-runtime-loop-check ]]; then
  if [[ "$MODE" == road-runtime-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/source_insert551_runtime_road.json")"; fi
fi
if [[ "$MODE" == road-runtime-diagnostic || "$MODE" == road-runtime-diagnostic-check ]]; then
  if [[ "$MODE" == road-runtime-diagnostic ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/diagnostic551_runtime_road_10s.json")"; fi
fi
if [[ "$MODE" == uncertainty-loop || "$MODE" == uncertainty-loop-check ]]; then
  if [[ "$MODE" == uncertainty-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/source_insert551_phase37_geometry.json")"; fi
fi
if [[ "$MODE" == diagnostic-loop || "$MODE" == diagnostic-loop-check ]]; then
  if [[ "$MODE" == diagnostic-loop ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/diagnostic551_memory_lifecycle_10s.json")"; fi
fi
if [[ "$MODE" == avoidance-validation || "$MODE" == avoidance-validation-check ]]; then
  if [[ "$MODE" == avoidance-validation ]]; then MODE=edited-closed-loop; else MODE=edited-closed-loop-check; fi
  if [[ "$#" -lt 2 ]]; then set -- "$1" "$(hugsim_source_path "$BRIDGE/closed_loop_integration/examples/avoidance551_18s.json")"; fi
fi
if [[ "$MODE" == edited-closed-loop-postprocess ]]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  exec "$PY" -B "$(hugsim_source_path "$BRIDGE/closed_loop_integration/stage_results.py")" --run "${2:?existing run directory required}"
fi
if [[ "$MODE" == edited-closed-loop-check ]]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  exec "$PY" -B "$(hugsim_source_path "$BRIDGE/closed_loop_integration/run_edited_closed_loop.py")" --check --task "${2:-$BRIDGE/closed_loop_integration/examples/source_insert551_memory_lifecycle.json}"
fi
if [[ "$MODE" == status ]]; then
  exec "$PY" -B "$(hugsim_source_path "$HERE/status.py")"
fi
if [[ "$MODE" == weather-check ]]; then
  exec "$PY" -B "$(hugsim_source_path "$HERE/weather_check.py")"
fi
if [[ "$MODE" == edit-check ]]; then
  exec "$PY" -B "$(hugsim_source_path "$HERE/edit_api.py")" "${2:?command JSON required}"
fi
if [[ "$MODE" == agent-json-check ]]; then
  exec "$PY" -B "$(hugsim_source_path "$BRIDGE/agent_json/run_agent_json.py")" \
    --job "${2:-$BRIDGE/agent_json/examples/edit_preview.json}" --check-only
fi
if [[ "$MODE" != edited-closed-loop && "$MODE" != run && "$MODE" != difix && "$MODE" != weather && "$MODE" != weather-window && "$MODE" != weather-causal && "$MODE" != weather-mask && "$MODE" != weather-mask-diagnose && "$MODE" != weather-rgb-trace && "$MODE" != weather-loop && "$MODE" != edited-weather-loop && "$MODE" != unified-edit-weather-loop && "$MODE" != managed-asset-weather-loop && "$MODE" != native-edit-completion && "$MODE" != scene-preview && "$MODE" != resident-validation && "$MODE" != resident-driving && "$MODE" != native-rotate-weather-loop && "$MODE" != render-quality && "$MODE" != render-quality-followup && "$MODE" != asset-quality && "$MODE" != quality-asset-weather-loop && "$MODE" != agent-json && "$MODE" != agent-json-stdio ]]; then
  echo 'Usage: run_phase26.sh search-loop|search-loop-check|search-diagnostic|search-diagnostic-check|envelope-loop|envelope-loop-check|envelope-diagnostic|envelope-diagnostic-check|memory-loop|memory-loop-check|memory-diagnostic|memory-diagnostic-check|road-runtime-loop|road-runtime-loop-check|road-runtime-diagnostic|road-runtime-diagnostic-check|uncertainty-loop|uncertainty-loop-check|diagnostic-loop|diagnostic-loop-check|edited-closed-loop-postprocess|edited-closed-loop|edited-closed-loop-check|status|weather-check|run|weather-loop|edited-weather-loop|unified-edit-weather-loop|managed-asset-weather-loop|native-edit-completion|scene-preview|resident-validation|resident-driving|native-rotate-weather-loop|render-quality|render-quality-followup|asset-quality|quality-asset-weather-loop|agent-json|agent-json-check|agent-json-stdio|weather-mask|weather-mask-diagnose [NEW_DIR]; weather-rgb-trace NEW_DIR [OPTIONS]; difix --input PNG --output NEW_DIR; weather --output NEW_DIR; weather-window NEW_DIR; weather-causal INPUTS NEW_DIR' >&2; exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMPL_NEUTRAL_NPZ="$(hugsim_source_path "$BRIDGE/smpl_tools/resources/smpl_neutral/SMPL_NEUTRAL.npz")"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$(hugsim_source_path "$BRIDGE/env_fix/torch_ext_isolated")"
LIBDIR=$(ls -d "$BRIDGE"/env_fix/nvdiffrast_build_src/build/lib.linux-*)
export PYTHONPATH="$BRIDGE/framework_backend:$BRIDGE/../framework_repo:$BRIDGE/env_fix/nvdiffrast_build_src:$LIBDIR:$BRIDGE/env_fix/nvdiffrast_meta:${PYTHONPATH:-}"
export PYTHONPATH="$(hugsim_pythonpath "$PYTHONPATH")"
# Hold the service lock for the entire chain, not just a transient availability check.
exec 9<"$(hugsim_source_path "$ROOT/outputs/gpu_test_service/worker.lock")"
flock -n -x 9 || { echo 'GPU service lock busy; no work started' >&2; exit 1; }
if pgrep -f '[r]un_arm.*py|[p]aired_closed_loop|[r]ender_edit.py|[v]alidate_articulation.py|[v]alidate_dynamic_nodes.py|[r]ender_inputs.py|[a]pply_weather_window.py|[a]pply_weather_causal.py|[v]alidate_masks.py|[d]iagnose_rgb.py|[t]race_rgb_drift.py|[w]eather_smoke.py|[r]un_weather_loop.py|[w]eather_worker.py|[r]un_edit_completion.py|[r]un_resident_validation.py|[r]un_resident_driving.py|[r]un_render_quality.py|[r]un_render_followup.py|[r]un_asset_quality.py|[r]un_agent_json.py|[c]losed172_worker.py' >/dev/null; then
  echo 'Another render/closed-loop process is active; no work started' >&2; exit 1
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required; use status on CPU node"'
if [[ "$MODE" == edited-closed-loop ]]; then
  test ! -e "$(hugsim_source_path "$ROOT/outputs/gpu_test_service/STOP")" || { echo 'Service STOP present' >&2; exit 1; }
  ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ "$ACTIVE" =~ [0-9] ]]; then echo 'GPU already has compute processes' >&2; exit 1; fi
  TASK="${2:-$BRIDGE/closed_loop_integration/examples/source_insert551_memory_lifecycle.json}"
  OUT="${3:-$BRIDGE/closed_loop_integration/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" && test ! -e "$OUT.log" || { echo "Refusing overwrite: $OUT" >&2; exit 1; }
  printf 'Edited scene -> frozen UniAD/event-MPC with geometry acceleration; output: %s\n' "$OUT"
  DRIVE_LIMIT=$("$PY" -B "$(hugsim_source_path "$BRIDGE/closed_loop_integration/run_edited_closed_loop.py")" --print-wall-limit --task "$TASK")
  set +e
  timeout --signal=INT --kill-after=30s "${DRIVE_LIMIT}s" "$PY" -u -B "$(hugsim_source_path "$BRIDGE/closed_loop_integration/run_edited_closed_loop.py")" --drive-only --task "$TASK" --output "$OUT" 2>&1 | tee "$OUT.log"
  DRIVE_EXIT=${PIPESTATUS[0]}
  set -e
  # Driving has ended; CPU presentation has its own bounded deadlines.
  flock -u 9
  exec 9<&-
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/closed_loop_integration/stage_results.py")" --run "$OUT" --simulation-exit "$DRIVE_EXIT" 2>&1 | tee "$OUT.postprocess.log"
  exit
fi
if [[ "$MODE" == agent-json ]]; then
  JOB_JSON="${2:-$BRIDGE/agent_json/examples/edit_preview.json}"
  OUT="${3:-$HERE/agent_json_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'JSON editing-agent job: %s\nOutput: %s\n' "$JOB_JSON" "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/agent_json/run_agent_json.py")" --job "$JOB_JSON" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == agent-json-stdio ]]; then
  OUT="${2:-$HERE/agent_json_streams/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.stderr.log" || { echo "refusing to overwrite: $OUT.stderr.log" >&2; exit 1; }
  printf 'Persistent JSONL editing-agent owner; output: %s; model logs: %s.stderr.log\n' "$OUT" "$OUT" >&2
  exec "$PY" -u -B "$(hugsim_source_path "$BRIDGE/agent_json/run_agent_json.py")" --jsonl --output "$OUT" 2>"$OUT.stderr.log"
fi
if [[ "$MODE" == asset-quality ]]; then
  OUT="${2:-$HERE/asset_quality_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Asset view/quality comparisons: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_asset_quality.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == render-quality-followup ]]; then
  OUT="${2:-$HERE/render_quality_followups/$(date -u +%Y%m%dT%H%M%SZ)}"
  SOURCE_RUN="${3:-$HERE/render_quality_runs/20261103T072234Z}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Render follow-up: 3 cases plus class layers; output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_render_followup.py")" --source-run "$SOURCE_RUN" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == render-quality ]]; then
  OUT="${2:-$HERE/render_quality_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Rendering quality repair and layer comparisons: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_render_quality.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == resident-driving ]]; then
  OUT="${2:-$HERE/resident_driving_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  STEPS="${3:-5}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  test -x "$(hugsim_source_path "$ROOT/tools/pixi-uniad-legacy/.pixi/envs/default/bin/python")"
  "$PY" -B "$(hugsim_source_path "$HERE/run_resident_driving.py")" --output "$OUT" --steps "$STEPS" --check-only
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Resident driving output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_resident_driving.py")" --output "$OUT" --steps "$STEPS" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == resident-validation ]]; then
  OUT="${2:-$HERE/resident_validation_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  test -x "$(hugsim_source_path "$ROOT/tools/pixi-uniad-legacy/.pixi/envs/default/bin/python")"
  "$PY" -B "$(hugsim_source_path "$HERE/run_resident_validation.py")" --output "$OUT" --check-only
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'New resident UniAD / scene switch validation output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_resident_validation.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == scene-preview ]]; then
  OUT="${2:-$HERE/scene_preview_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  STEPS="${3:-50}"
  BODY_RUN="$(hugsim_source_path "$HERE/edited_weather_runs/20261102T031621Z")"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  "$PY" -c 'import sys; n=int(sys.argv[1]); assert 1 <= n <= 117, "steps must be 1..117"' "$STEPS"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  mkdir -p "$(dirname "$OUT")"
  mkdir "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  printf 'New scene preview: %s controls at 10Hz; output: %s\n' "$STEPS" "$OUT"
  "$PY" -B "$(hugsim_source_path "$BRIDGE/fast_control/checkpoint_preflight.py")" --output "$(hugsim_source_path "$OUT/checkpoint_preflight.json")"
  "$PY" -B "$(hugsim_source_path "$HERE/reuse_articulation.py")" --run "$BODY_RUN" --output "$(hugsim_source_path "$OUT/articulation_reuse.json")"
  RESULT=0
  if "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/run_weather_loop.py")" --output "$(hugsim_source_path "$OUT/joint")" \
    --steps "$STEPS" --edit-command "$(hugsim_source_path "$HERE/native_rigid_command.json")" --verify-edit-lifecycle \
    --articulation-report "$(hugsim_source_path "$BODY_RUN/articulation/report.json")" 2>&1 | tee "$(hugsim_source_path "$OUT/run.log")"; then
    RESULT=0
  else
    RESULT=$?
  fi
  if [[ -f "$(hugsim_source_path "$OUT/joint/weather/report.json")" && -f "$(hugsim_source_path "$OUT/joint/closed_loop/run_report.json")" ]]; then
    "$PY" -u -B "$(hugsim_source_path "$HERE/present_scene.py")" --run "$(hugsim_source_path "$OUT/joint")" --output "$(hugsim_source_path "$OUT/presentation")"
  fi
  exit "$RESULT"
fi
if [[ "$MODE" == native-edit-completion ]]; then
  OUT="${2:-$HERE/native_completion_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  "$PY" -B "$(hugsim_source_path "$HERE/edit_api.py")" "$(hugsim_source_path "$HERE/native_rigid_command.json")"
  "$PY" -B "$(hugsim_source_path "$HERE/edit_api.py")" "$(hugsim_source_path "$HERE/native_deformable_command.json")"
  SWITCH_ARGS=()
  if [[ -n "${PHASE26_SWITCH_MANIFEST:-}" ]]; then
    SWITCH_ARGS=(--switch-manifest "$PHASE26_SWITCH_MANIFEST")
  fi
  printf 'Native completion output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$HERE/run_edit_completion.py")" --output "$OUT" "${SWITCH_ARGS[@]}"
  exit
fi
if [[ "$MODE" == managed-asset-weather-loop || "$MODE" == quality-asset-weather-loop ]]; then
  if [[ "$MODE" == quality-asset-weather-loop ]]; then
    OUT="${2:-$HERE/quality_asset_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
    COMMAND="${3:-$HERE/managed_vehicle_quality_command.json}"
  else
    OUT="${2:-$HERE/managed_asset_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
    COMMAND="${3:-$HERE/managed_vehicle_command.json}"
  fi
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  "$PY" -B "$(hugsim_source_path "$HERE/edit_api.py")" "$COMMAND"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  mkdir "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  printf 'Managed asset integration output: %s\n' "$OUT"
  "$PY" -B "$(hugsim_source_path "$HERE/fingerprint.py")" "$OUT"
  "$PY" -B "$(hugsim_source_path "$HERE/reuse_articulation.py")" --run "$(hugsim_source_path "$HERE/edited_weather_runs/20261102T031621Z")" --output "$(hugsim_source_path "$OUT/articulation_reuse.json")"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/run_weather_loop.py")" --output "$(hugsim_source_path "$OUT/joint")" \
    --edit-command "$COMMAND" --verify-edit-lifecycle 2>&1 | tee "$(hugsim_source_path "$OUT/joint.log")"
  exit
fi
if [[ "$MODE" == unified-edit-weather-loop || "$MODE" == native-rotate-weather-loop ]]; then
  if [[ "$MODE" == native-rotate-weather-loop ]]; then
    OUT="${2:-$HERE/native_rotation_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
    COMMAND="${3:-$HERE/native_rigid_rotation_command.json}"
  else
    OUT="${2:-$HERE/unified_edit_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
    COMMAND="${3:-$HERE/native_pedestrian_command.json}"
  fi
  BODY_RUN="${PHASE26_BODY_EVIDENCE_RUN:-$HERE/edited_weather_runs/20261102T031621Z}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  "$PY" -B "$(hugsim_source_path "$HERE/edit_api.py")" "$COMMAND"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  mkdir "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  printf 'Unified edit integration output: %s\n' "$OUT"
  "$PY" -B "$(hugsim_source_path "$HERE/fingerprint.py")" "$OUT"
  "$PY" -B "$(hugsim_source_path "$HERE/reuse_articulation.py")" --run "$BODY_RUN" --output "$(hugsim_source_path "$OUT/articulation_reuse.json")"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/run_weather_loop.py")" --output "$(hugsim_source_path "$OUT/joint")" \
    --edit-command "$COMMAND" --verify-edit-lifecycle \
    --articulation-report "$(hugsim_source_path "$BODY_RUN/articulation/report.json")" 2>&1 | tee "$(hugsim_source_path "$OUT/joint.log")"
  exit
fi
if [[ "$MODE" == edited-weather-loop ]]; then
  OUT="${2:-$HERE/edited_weather_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  mkdir "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  printf 'Edited weather integration output: %s\n' "$OUT"
  "$PY" -B "$(hugsim_source_path "$HERE/fingerprint.py")" "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/smpl_dynamic/validate_articulation.py")" --output "$(hugsim_source_path "$OUT/articulation")" 2>&1 | tee "$(hugsim_source_path "$OUT/articulation.log")"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/run_weather_loop.py")" --output "$(hugsim_source_path "$OUT/joint")" \
    --native-edit "$(hugsim_source_path "$HERE/native_pedestrian_edit.json")" \
    --articulation-report "$(hugsim_source_path "$OUT/articulation/report.json")" 2>&1 | tee "$(hugsim_source_path "$OUT/joint.log")"
  exit
fi
if [[ "$MODE" == weather-loop ]]; then
  OUT="${2:-$HERE/weather_closed_loop_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  test -x "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" || { echo 'Weather environment missing' >&2; exit 1; }
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/run_weather_loop.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == weather-rgb-trace ]]; then
  OUT="${2:?new trace output directory required}"
  shift 2
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/trace_rgb_drift.py")" --output "$OUT" "$@" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == weather-mask-diagnose ]]; then
  OUT="${2:-$HERE/weather_mask_diagnostics/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Mask RGB diagnostic output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/diagnose_rgb.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == weather-mask ]]; then
  OUT="${2:-$HERE/weather_mask_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname "$OUT")"
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  printf 'Mask validation output: %s\n' "$OUT"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_runtime/validate_masks.py")" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == weather-causal ]]; then
  INPUTS="${2:?existing measured inputs required}"
  OUT="${3:?new output directory required}"
  test -f "$(hugsim_source_path "$INPUTS/manifest.json")"
  mkdir -p "$(dirname "$OUT")"
  # Python owns creation; tee stores its log next to the new output directory.
  test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  test ! -e "$OUT.log" || { echo "refusing to overwrite: $OUT.log" >&2; exit 1; }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" \
    -u -B "$(hugsim_source_path "$BRIDGE/weather_sequence/apply_weather_causal.py")" --inputs "$INPUTS" --output "$OUT" 2>&1 | tee "$OUT.log"
  exit
fi
if [[ "$MODE" == weather-window ]]; then
  OUT="${2:?new output directory required}"
  mkdir -p "$(dirname "$OUT")"
  mkdir "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  "$PY" -u -B "$(hugsim_source_path "$BRIDGE/weather_sequence/render_inputs.py")" --output "$(hugsim_source_path "$OUT/inputs")" 2>&1 | tee "$(hugsim_source_path "$OUT/render.log")"
  # Separate processes release renderer memory and avoid generic module-name collisions.
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$(hugsim_source_path "$BRIDGE/env_fix/weather_env_v1/bin/python")" \
    -u -B "$(hugsim_source_path "$BRIDGE/weather_sequence/apply_weather_window.py")" --inputs "$(hugsim_source_path "$OUT/inputs")" --output "$(hugsim_source_path "$OUT/weather")" 2>&1 | tee "$(hugsim_source_path "$OUT/weather.log")"
  exit
fi
if [[ "$MODE" == weather ]]; then
  shift
  "$PY" -B "$(hugsim_source_path "$HERE/weather_smoke.py")" "$@"
  exit
fi
if [[ "$MODE" == difix ]]; then
  shift
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -B "$(hugsim_source_path "$HERE/difix_smoke.py")" "$@"
  exit
fi
OUT="${2:-$HERE/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
# Output path becomes part of acceptance evidence; never reuse it.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
"$PY" -B "$(hugsim_source_path "$HERE/fingerprint.py")" "$OUT"
"$PY" -u -B "$(hugsim_source_path "$BRIDGE/smpl_dynamic/validate_articulation.py")" --output "$(hugsim_source_path "$OUT/articulation")"
# Validator exits nonzero unless both node groups pass. No closed loop on failure.
"$PY" -u -B "$(hugsim_source_path "$BRIDGE/framework_backend/run_arm_native_edit.py")" \
  "$(hugsim_source_path "$BRIDGE/framework_backend/requests/fastprobe_fwbackend.json")" \
  "$(hugsim_source_path "$HERE/native_pedestrian_edit.json")" "$(hugsim_source_path "$OUT/edited_closed_loop")"
"$PY" -B "$(hugsim_source_path "$HERE/accept_run.py")" "$OUT"
