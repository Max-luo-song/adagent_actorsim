# v11 risk provenance repair

Independent direct B release. v10 sources, observations and results are preserved.

Root cause: candidate evaluation called observe_conflicts() for every proposed
bypass. A freshly detected peripheral object touched by a rejected hypothetical
route acquired a persistent risk latch. This changed later obstacle memory,
even when the current neural intention and actual executed route never touched
that object. Ordinary inactive identity expiry was then disabled indefinitely.

Only current neural intention (native prefix and its centerline preview) can
establish that persistent obligation. Side candidates still check the same
complete current geometry, but cannot change persistent memory. Confirmed
on-intent and occluded latched obstacles retain the existing lifecycle and all
hard collision checks. Inactive unlatched expiry is unchanged and makes no
physical free-space claim. No ID, class or scene suppression is used.

The motion warm-up threshold, geometry margins, road polygon, preview horizon,
collision certificates, original iLQR, native collision optimization, model,
renderer, scenario and physical bounds are unchanged. A short-window motion
threshold ablation did NOT repair the actual failure and was not adopted.

Tests reproduce the old bug under identical inputs, verify unchanged current
geometry/input arrays and genuine side-on-intent risk retention, and replay all
188 saved v10 frames. Existing multi-actor/noisy continuous actual-iLQR tests
remain required. CPU replay and synthetic observation rollouts do not certify a
new rendered GPU outcome. The fresh direct B is required to measure that.

bash run.sh full runs 188 controls / 18.8s from fresh inference, with v10 used
only for post-hoc comparison. No new A and no archived controls are replayed.
