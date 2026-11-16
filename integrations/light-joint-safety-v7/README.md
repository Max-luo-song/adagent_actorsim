# Joint road and retained-actor safety gate v7

Fixes the B-v6 failure chain without changing model weights, iLQR, scene edit,
physical bounds or the authorized white-edge polygon. No heavy MPC is added.

- A legal native avoidance reference is preserved exactly. Only out-of-polygon
  lateral nodes are projected; the resulting complete footprint is rechecked.
- Retained detections are associated geometrically across ego poses. Duplicate
  rows are merged. At least three observations spanning 0.19s support a track;
  velocity is fitted from world displacement, not copied from neural velocity.
- Positive predicted conflicts latch risk. Missing occupancy/detections cannot
  clear a latch or renew its evidence timestamp. Release needs three current
  observations with separated reference and admissible stopping trajectory.
- Every action passes both road and actor stopping-path checks. Two bounded
  braking witnesses (unwind and hold steering) are available. No valid witness
  yields an explicit constraint conflict, never fabricated clearance.

The margins and retained-track model are conservative engineering hypotheses,
not a certified physical sensor envelope. A new single-frame object may not
have confirmed memory yet. Native occupancy avoidance remains active.
If the model offers no legal obstacle-free bypass, the gate can stop and wait;
this release does not promise completion of the maneuver.

Per the user's latest instruction, GPU `full` runs only edited B, starting
from fresh inference/control at observation0 with complete model history.
No old A actions are replayed. Old B is an outcome comparison, not a matched
per-frame causal experiment. A new normal-scene closed-loop regression is
not performed in this run. Saved A CPU replay remains a limited check.
Scores are post-hoc diagnostics, never controller inputs.
Old releases and all completed outputs remain untouched.

Run `bash run.sh full` on an idle GPU node after CPU admission/prepare/check.
Default output: phase26/light_joint_safety_v7. Existing partial arms are retained
for diagnosis; do not overwrite them or call them completed.
