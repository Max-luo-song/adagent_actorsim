"""closed172_common substitute: iLQR tracking passthrough instead of evidence-tiered MPC.

DIAGNOSTIC, UNCERTIFIED. This module drops ALL obstacle-avoidance optimization:
occupancy, reliability reports, drivable contracts and guard constraints are
accepted for signature compatibility but NEVER evaluated against the selected
controls. It exists so architecture-integration work (weather, dynamic objects,
sensors, full-scene loading) can run the closed loop without waiting 77-409s
per step for the MPC. It does NOT provide the MPC's collision-avoidance
capability and MUST NOT be claimed as safety-equivalent. The original
closed172_common (and the validated opt copy in bridge/opt/) remain unchanged
as the full-control reference.

Control path: UniAD plan (6x2, [right, forward], ego planning frame)
-> waymo23-native-import/trajectory_control.reference_states
-> HUGSIM sim/ilqr/lqr.plan2control (iLQR, max_solve_time 0.05s per solve)
-> clip to shared vehicle limits (scene_api.validate_action contract)
-> 6x2 control sequence (per-substep re-solve rollout, lqr-local frame)
-> RecoveryController applies row 0 (unchanged state machine).

Fallback semantics: an invalid plan or nonfinite solver output yields
selected=None, which makes the reviewed RecoveryController brake to a stop
(state 'braking'/'stopped_reassessing'). That stop is a REASSESSMENT fallback,
not a certified-safe action: without obstacle evaluation there is no claim
that braking avoids collision.
"""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import importlib.util
import json
import math
import sys
import time
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent            # bridge/fast_control
JOB = HERE.parents[3]                             # unified-open-closed-172-20261022-v4
P11 = JOB / 'chain_repair/phase11'
ROOT = JOB.parents[3]
for _p in (str(P11), str(ROOT / 'integrations/sr-hugsim'),
           str(ROOT / 'waymo23-native-import'),
           str(ROOT / 'waymo23-native-import/waymo172_closed_loop_v1'),
           str(ROOT / 'code/HUGSIM')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location('closed172_common_reviewed', P11 / 'closed172_common.py')
_real = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _real
_spec.loader.exec_module(_real)
assert Path(_real.__file__).resolve() == (P11 / 'closed172_common.py').resolve()

# Re-export EVERY public name from the reviewed module (not only the names the
# harness imports directly): native_geometry_evaluation / native_actor_evaluation
# do `from closed172_common import polygon_check` at call time, dynamic_completion
# imports body_vertices/asset_reference_vertices, etc. A whitelist here caused the
# 2026-10-29 step-0 launch crash (ImportError on polygon_check); the failure site
# is preserved as closed_arms/fastprobe_A5_tracking_v1.launch_failed_20261029.
for _name in dir(_real):
    if not _name.startswith('_') and _name != 'optimize':
        globals()[_name] = getattr(_real, _name)
del _name

TRACKING_POLICY = 'tracking_passthrough_no_obstacle_optimization_v1'


def _default_control_interval_s():
    cfg = _hugsim_expand(json.loads((ROOT / 'waymo23-native-import/configs/planning.json').read_text()))
    return float(cfg['control_interval_s'])


def _clip_action(acc, rate, speed, steer, vehicle):
    """Clip (acc, steer_rate) so scene_api.validate_action cannot reject them."""
    dt = float(vehicle.dt)
    margin = 1e-9
    lo = max(float(vehicle.min_acc_mps2), -max(0.0, speed) / dt) + margin
    hi = min(float(vehicle.max_acc_mps2), (float(vehicle.max_speed_mps) - speed) / dt) - margin
    if lo > hi:  # conflicting state-derived bounds: fall back to physical limits
        lo, hi = float(vehicle.min_acc_mps2) + margin, float(vehicle.max_acc_mps2) - margin
    acc = min(max(float(acc), lo), hi)
    max_rate = math.radians(float(vehicle.max_steer_rate_degrees))
    max_steer = math.radians(float(vehicle.max_steer_degrees))
    rlo = max(-max_rate, (-max_steer - steer) / dt) + margin
    rhi = min(max_rate, (max_steer - steer) / dt) - margin
    rate = min(max(float(rate), rlo), max(rlo, rhi))
    return acc, rate


def _bicycle_step(state, acc, rate, dt, wheelbase):
    """Kinematic bicycle in the lqr-local frame (x=forward, y=right); matches
    sim/ilqr/lqr_solver.py's discrete model."""
    x, y, theta, v, delta = state
    v2 = max(0.0, v + acc * dt)  # vehicle cannot reverse (validate_action: speed>=0)
    delta2 = delta + rate * dt
    return np.array([
        x + v * math.cos(theta) * dt,
        y + v * math.sin(theta) * dt,
        theta + v * math.tan(delta) / wheelbase * dt,
        v2, delta2])


def optimize(reference, occupancy, info, warm=None, maxiter=None, reliability=None,
             vehicle=None, control_interval_s=None, _forecast_recovery=False,
             _reuse_verified_warm=False):
    """Drop-in replacement for closed172_common.optimize: iLQR tracking only.

    `occupancy`, `reliability`, `warm`, `maxiter` are accepted but unused for
    constraint purposes (recorded as ignored). Returns the same dict shape the
    harness and RecoveryController consume; `selected.feasible` is None and
    `minimum_constraint` is None because NO constraint was evaluated.
    """
    t_start = time.perf_counter()
    vehicle = vehicle or load_vehicle_config()
    audit = dict(control_path=TRACKING_POLICY,
                 obstacle_optimization='none: occupancy/reliability/guard constraints not evaluated',
                 safety_equivalence_to_mpc=False,
                 mpc_reference='phase11 closed172_common.optimize (unchanged); bridge/opt optimized copy',
                 recovery=dict(confirmation_seconds=.2, confirmation_observations=1, max_gap_seconds=.75),
                 evidence_tiers=[])
    base = dict(trials=[], forecast_recovery=False,
                occupancy_constraint_policy='ignored (tracking passthrough)',
                planning_horizon_s=None, neural_horizon_s=3.,
                guard=audit, ground_truth_used_by_optimizer=False,
                policy='iLQR tracking of the fixed UniAD plan; no obstacle avoidance optimization; '
                       'fallback braking is a reassessment stop, not certified safe',
                steering_physical_limit_validated=False)
    try:
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2 \
                or not np.isfinite(reference).all():
            return dict(base, selected=None, reason='tracking_invalid_model_trajectory',
                        solve_seconds=time.perf_counter() - t_start)
        ci = float(control_interval_s) if control_interval_s is not None else _default_control_interval_s()
        if not np.isfinite(ci) or ci <= 0 or not np.isclose(ci / vehicle.dt, round(ci / vehicle.dt)):
            raise ValueError('Invalid planning control interval')
        speed, steer = float(info['ego_velo']), float(info['ego_steer'])
        if not np.isfinite([speed, steer]).all() or speed < 0:
            raise ValueError('Invalid ego motion state')

        from trajectory_control import reference_states  # waymo23-native-import (unchanged)
        from sim.ilqr.lqr import plan2control              # code/HUGSIM (unchanged)

        ref_states = reference_states(reference)
        state = np.array([0., 0., 0., speed, steer], dtype=float)
        controls = np.zeros((6, 2))
        states = [state.copy()]
        n_clipped = 0
        for k in range(6):
            acc, rate = plan2control(ref_states, state)
            if not np.isfinite([acc, rate]).all():
                return dict(base, selected=None, reason='tracking_solver_nonfinite_output',
                            solve_seconds=time.perf_counter() - t_start)
            acc_c, rate_c = _clip_action(acc, rate, state[3], state[4], vehicle)
            n_clipped += int((abs(acc_c - acc) > 0) or (abs(rate_c - rate) > 0))
            controls[k] = (acc_c, rate_c)
            state = _bicycle_step(state, acc_c, rate_c, ci, float(vehicle.wheelbase_m))
            states.append(state.copy())
        states = np.asarray(states)
        selected = dict(
            seed='ilqr_tracking_rollout', solver_success=True, optimization_skipped=False,
            message='iLQR tracking of UniAD plan only; no obstacle constraints evaluated',
            feasible=None, minimum_constraint=None,
            seed_name_is_final_direction=False,
            lateral_extent_m=float(np.abs(states[:, 1]).max()) if len(states) else 0.,
            clearance_lower_bound_m=None, objective=None,
            used_verified_seed=False, warm_reference_match=None, warm_reference_max_deviation_m=None,
            controls=controls.tolist(), states=states.tolist(),
            states_frame='lqr_local: x=forward, y=right, heading from forward axis; '
                         'rows 1-5 are tracking rollout extrapolation, only row 0 is applied',
            lidar=None, control_interval_s=ci,
            n_vehicle_limit_clips=n_clipped,
            tracking_rollout_solves=6)
        return dict(base, selected=selected, reason=None,
                    solve_seconds=time.perf_counter() - t_start)
    except Exception as exc:
        # A tracking-side failure must look like a failed plan to the reviewed
        # RecoveryController (brake/reassess), never like a feasible plan.
        return dict(base, selected=None, reason='tracking_internal_error: %s' % exc,
                    solve_seconds=time.perf_counter() - t_start)
