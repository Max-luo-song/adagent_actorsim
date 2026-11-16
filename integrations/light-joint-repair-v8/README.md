# Joint corridor and velocity repair v8

Direct edited B only, per user instruction. Old v7 files/results remain frozen.
The existing UniAD/model/renderer/actor edit, white-edge polygon, iLQR solver
and vehicle bounds are retained. No heavy multistart MPC or native-score
objective is introduced. Native scoring runs only after execution.

The repair preserves legal reachable native references with no unresolved
front conflict. Otherwise it samples at most 15 combinations of road/actor
corridors and speed targets. Each candidate is integrated with the supplied
vehicle update order and bounded acceleration, steering and steering rate.
Preview is bounded to 5--12 seconds, with conservative geometric uncertainty
beyond the supported 2.5s forecast. Complete footprints, inter-sample motion,
all retained actors, physical bounds and terminal braking are checked jointly.
No candidate means verified braking/waiting, never fabricated free space.

Candidates use bounded geometric guidance only to construct physical rollouts;
the ACTUAL control remains the same iLQR solver. Its new adapter supplies the
31 physical reference states directly, preserving position, yaw, velocity and
steering, instead of collapsing the path to six points and reconstructing it.
The six half-second points remain the native-score/plot export. A current-road
parallel seed permits heading recovery; it is not a frozen initial lane anchor.

Risk timestamps survive missing queries. A stationary-compatible world track
cannot release risk merely due to a fitted velocity sign change. Positive
acceleration is bounded by the selected physical speed profile. The final gate
checks the actual first action with either a safe immediate stop or a verified
tracking continuation ending in a stop. A lane change may continue alongside
an object only with a valid joint witness. These are engineering hypotheses,
not calibrated physical sensor certification. No simulator actor truth or
recorded future ego trajectory enters the controller.

CPU validation includes saved B-v7 failure frames, saved normal A, mirrored
roads, blocked roads, missing queries, multiple initial speeds, dense-reference
transport, exact native Motion, and continuous actual iLQR with synthetic sensor
observations. This is not a rendered GPU closed loop or a formal score claim.
GPU full runs ONLY B for the original 188 controls / 18.8 seconds.

Run `bash run.sh full` after CPU admission, prepare and check. It records raw
native trajectories, selected dense rollouts, every candidate rejection,
executed controls, native scores and fully decoded v7/v8 comparison video.
