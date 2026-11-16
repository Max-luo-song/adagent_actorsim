# v10 observation evidence and continuous control

Independent direct-B release; v9 source/results remain immutable. Models,
renderer, actor edit, road polygon, original iLQR solver and physical vehicle
bounds are unchanged. Native scores remain post-hoc.

Observed geometry and unobserved uncertainty are separate. Every confirmed
neural object's observed core remains collision-checked, including pedestrians
and stale locked obstacles. Its size follows measured footprint/dispersion,
never elapsed-time inflation. Age uncertainty is a bounded, confidence-weighted
risk cost; it cannot manufacture a ten-meter physical object. Height support
is audited against the declared model-ground reference, not treated as proof
that an object is absent. No scene/track IDs are suppressed.

Motion uses positive observations in a bounded 1.6s robust window. Position
innovations, direction consistency, displacement versus range-dependent noise
and fit residuals distinguish supported motion from an unresolved hypothesis
including stationarity. Relocalization starts a new motion segment. Coherent
approaching/receding objects retain their motion. No future actor truth is read.

Candidate speeds include the current neural intention, a current-problem warm
speed and a slower option. Cross-frame target and corridor switching costs
reduce flicker; old targets cannot override a new neural stop. Nearby blockers
are grouped into a single bypass segment. Bypass lanes follow CURRENT road
orientation in the rotating ego frame and reconnect to current neural intent,
not an initial-frame anchor. At most 15 bounded candidate combinations run.

Ordinary executed acceleration is constrained on BOTH sides by the jerk range
and the selected profile. The real iLQR action is then checked jointly for road,
actors, dynamics and a stopping/continuation witness. When no tracking witness
exists, a smooth protective transition is attempted before maximum braking.
Only a verified emergency may override comfort; speed saturation is explicitly
audited. Actual saturated rollout commands match native Motion integration.

CPU coverage includes actual v9 false approach/stall frames, long-lived weak
pedestrian geometry, real occluded pedestrians, genuine oncoming/fast vehicles,
ordinary/emergency control, saturation, noisy-car/transient-detection continuous
actual iLQR, plus v9 regressions. Synthetic/replayed evidence is NOT a rendered
GPU outcome or physical sensor certification. Risk/covariance assumptions
remain explicit engineering hypotheses; safe waiting remains possible.

`bash run.sh full` runs only B for 188 controls / 18.8s, from fresh inference.
It exports raw/current/adopted plans, dense physical references, candidate
rejections, evidence provenance, executed controls, native metrics and a v9/v10
comparison video. See REPORT.md and checks/ before GPU execution.
