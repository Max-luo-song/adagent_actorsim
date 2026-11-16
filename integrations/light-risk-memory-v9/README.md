# v9 per-candidate risk memory

Fixes v8's frame23 global stale-risk veto and its frame43 permanent stop.
Only direct B runs on GPU. Old v8 sources and outputs remain immutable.

A stale identity is never an independent veto. Every candidate still checks
all retained actor geometry, age-driven uncertainty, authorized road boundaries,
physical bounds, full footprints and terminal stopping. Actual iLQR actions
still require a safe stop or safe continuation ending in a stop.

Only current positive observations plus geometric conflict create risk locks.
Age-driven expansion never creates a new lock. An observed refreshed track
with positive current-reference separation may release its old lock, while its
current geometry remains checked. Inactive unlatched identities expire without
claiming observed free space. Unresolved locked obstacles survive missing
queries. Long-gap shape-compatible reacquisition and overlap-based stale-alias
merging prevent old identities from living alongside their refreshed copies.
Evidence timestamps are renewed only by current detections, not by replay.
Fresh-frame deduplication, association and world-velocity fitting preserve the
reviewed v8 behavior; this release changes stale lifecycle rather than model
weights or thresholds. Lost motion uses the declared constant-world-velocity
and growing-position-margin hypothesis; this is not physical certification.

Stopping witnesses saturate at zero speed and steering bounds at every step.
Repeating braking during the reaction interval no longer predicts reverse
motion and spuriously rejects a nearly stopped vehicle.

CPU tests include the entire real v8 failed history, actual iLQR controls at
frame23 and frame44, occluded real obstacles on a blocked road, bypasses around
retained stale geometry, long-gap reacquisition, duplicate identity cleanup,
no stale-only lock promotion, near-zero stopping, multiple-actor dropout with
actual iLQR and vehicle integration, and the previous v8 regression suite.
Saved-frame replay and synthetic observations are NOT a rendered GPU outcome.

`bash run.sh full` runs only B for 188 controls, exports native scores and a
fully decoded v8/v9 comparison video. Scores remain post-hoc and are not
controller inputs. Review REPORT.md and checks/ before running on CUDA.
