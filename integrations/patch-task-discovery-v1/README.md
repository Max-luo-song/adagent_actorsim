# Background asset and off/on scene probes

Use `integrations/autonomous-patch-agent-v7/background.sh`:

- `start-assets`: detach a durable asset worker; CPU geometry/calibration/raw identity and road-support export run immediately for 172/552/621; semantic GPU and point transfer wait for GPU. Road-support geometry retains holes and is not represented as an independently verified white-line map. Scene runtime/map admission remains separate.
- `start-pair`: detach one four-episode original/edited × native/off/external/on comparison. Uses the highest measured complete branch selected from the persisted registry (currently v15), preserving the stable alternative in the contract. Shared native collision optimization, model weights, seed, dt, tracker, scene, native scorer and termination remain fixed. No archived actions or planning-score surrogate.
- `brief`, `status`, `stop`, `resume-assets`, `resume-pair`.

Results live under `phase26/patch_task_discovery_v1`. `failure_reason.json`, `edit_instruction.json` and `pair/comparison.json` are automatic. Instances need the project disk retained and the background entry restarted after reboot; no process can survive instance shutdown. Old patch-agent budgets and frozen host files are unchanged. New preprocessing/diagnostic budgets are declared separately and not reset by restart.

A lower native baseline score and a higher patched score on complete matched episodes is an improvement candidate, even without a collision. Incomplete score prefixes are not directly subtracted from full episodes. Infrastructure failures, unknown-only collisions, score loss caused merely by legal yielding, and unverified edits are not silently promoted to control bugs. Improvement is measured rather than promised.

Current real closed-loop adapter supports only the existing admitted 551 original-vehicle cut-in program. Other scenes may export assets in parallel, but cannot enter native closed-loop validation before a matching runtime/map/evaluator adapter is admitted. The probe produces evidence and does not yet register a new autonomous patch task or alter the frozen v7 task queue. This is explicitly recorded as `next_task_ready=false`; it must not be advertised as an end-to-end automatic scene-to-patch transition.
