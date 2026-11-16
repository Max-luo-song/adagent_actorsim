"""CPU tests for the tracking-passthrough control variant (diagnostic, uncertified)."""

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

import json
import math
import pickle
import sys
import time
import unittest
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
JOB = HERE.parents[3]
P11 = JOB / 'chain_repair/phase11'
ROOT = JOB.parents[3]
for p in (str(HERE), str(P11), str(ROOT / 'integrations/sr-hugsim'),
          str(ROOT / 'waymo23-native-import'), str(ROOT / 'code/HUGSIM')):
    if p not in sys.path:
        sys.path.insert(0, p)

import tracking_common as tc
import planning_guard_tracking as pgt

VEHICLE = tc.load_vehicle_config()
A5 = JOB / 'chain_repair/phase26/bridge/closed_arms/fastprobe_A5_opt_v1/arm/uniad_worker'


def info_at(speed, steer=0.0):
    return dict(ego_velo=float(speed), ego_steer=float(steer))


def straight_plan(forward_step=1.0):
    plan = np.zeros((6, 2))
    plan[:, 1] = np.arange(1, 7) * forward_step
    return plan


class TestTrackingOptimize(unittest.TestCase):
    def test_straight_plan_finite_within_limits(self):
        out = tc.optimize(straight_plan(), None, info_at(2.0), vehicle=VEHICLE)
        self.assertIsNone(out['reason'])
        sel = out['selected']
        self.assertIsNotNone(sel)
        controls = np.asarray(sel['controls'])
        self.assertEqual(controls.shape, (6, 2))
        self.assertTrue(np.isfinite(controls).all())
        self.assertTrue(np.all(controls[:, 0] >= VEHICLE.min_acc_mps2 - 1e-9))
        self.assertTrue(np.all(controls[:, 0] <= VEHICLE.max_acc_mps2 + 1e-9))
        max_rate = math.radians(VEHICLE.max_steer_rate_degrees)
        self.assertTrue(np.all(np.abs(controls[:, 1]) <= max_rate + 1e-9))

    def test_curved_plan_turns(self):
        plan = straight_plan(1.0)
        plan[:, 0] = np.arange(1, 7) * 0.3  # rightward offset grows
        out = tc.optimize(plan, None, info_at(3.0), vehicle=VEHICLE)
        rate = out['selected']['controls'][0][1]
        self.assertTrue(np.isfinite(rate))
        self.assertNotEqual(rate, 0.0)

    def test_zero_plan_does_not_accelerate(self):
        out = tc.optimize(np.zeros((6, 2)), None, info_at(3.0), vehicle=VEHICLE)
        acc = out['selected']['controls'][0][0]
        self.assertLessEqual(acc, 1e-9)

    def test_invalid_plan_returns_no_selection(self):
        for bad in (np.full((6, 2), np.nan), np.zeros((6, 3)), np.zeros((1, 2))):
            out = tc.optimize(bad, None, info_at(1.0), vehicle=VEHICLE)
            self.assertIsNone(out['selected'])
            self.assertEqual(out['reason'], 'tracking_invalid_model_trajectory')

    def test_solver_failure_path_is_brake_not_fake_feasible(self):
        # selected=None must be the only failure surface; feasible must never
        # be claimed by this module.
        out = tc.optimize(np.full((6, 2), np.nan), None, info_at(1.0), vehicle=VEHICLE)
        self.assertIsNone(out['selected'])
        self.assertNotEqual(out.get('reason'), None)
        good = tc.optimize(straight_plan(), None, info_at(1.0), vehicle=VEHICLE)
        self.assertIsNone(good['selected']['feasible'])
        self.assertIsNone(good['selected']['minimum_constraint'])

    def test_clip_respects_speed_ceiling(self):
        out = tc.optimize(straight_plan(3.0), None,
                          info_at(VEHICLE.max_speed_mps), vehicle=VEHICLE)
        acc = out['selected']['controls'][0][0]
        self.assertLessEqual(VEHICLE.max_speed_mps + acc * VEHICLE.dt,
                             VEHICLE.max_speed_mps + 1e-9)

    def test_output_json_serializable(self):
        out = tc.optimize(straight_plan(), None, info_at(2.0), vehicle=VEHICLE)
        json.dumps(out)  # strict-enough: no numpy types, no NaN
        self.assertNotIn('NaN', json.dumps(out))

    def test_honest_labels(self):
        out = tc.optimize(straight_plan(), None, info_at(2.0), vehicle=VEHICLE)
        self.assertFalse(out['guard']['safety_equivalence_to_mpc'])
        self.assertIn('no obstacle', out['policy'])
        self.assertIn('not certified safe', out['policy'])

    def test_solve_time_budget(self):
        t0 = time.perf_counter()
        tc.optimize(straight_plan(), None, info_at(2.0), vehicle=VEHICLE)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0)  # iLQR budget: 6 x 0.05s solve + overhead


class TestRecoveryControllerLabeling(unittest.TestCase):
    def test_decision_relabeled(self):
        out = tc.optimize(straight_plan(), None, info_at(2.0), vehicle=VEHICLE)
        ctrl = pgt.RecoveryController()
        action, decision = ctrl.update(out, 2.0, 80.0, VEHICLE.dt,
                                       VEHICLE.emergency_braking_mps2)
        self.assertEqual(decision['reason'], 'tracking_passthrough_unverified')
        self.assertIn('control_path', decision)
        self.assertIn('NOT the evidence-tiered MPC', decision['policy'])
        self.assertEqual(action['acc'], out['selected']['controls'][0][0])

    def test_no_selection_brakes(self):
        out = tc.optimize(np.full((6, 2), np.nan), None, info_at(2.0), vehicle=VEHICLE)
        ctrl = pgt.RecoveryController()
        action, decision = ctrl.update(out, 2.0, 8.0, VEHICLE.dt,
                                       VEHICLE.emergency_braking_mps2)
        self.assertEqual(decision['state_after'], 'braking')
        self.assertLess(action['acc'], 0.0)


class TestOfflineReplayA5(unittest.TestCase):
    """Replay saved fastprobe A5 perceptions through the tracking path.

    Records solve time and the tracking-vs-MPC first-action difference.
    Comparison is observational only: the two control paths answer different
    questions (tracking vs constrained optimization).
    """

    def test_replay_saved_perceptions(self):
        rows = []
        for step in range(5):
            folder = A5 / f'step_{step:03d}'
            perception = folder / 'perception.npz'
            packet = folder / 'observation.pkl'
            if not perception.exists():
                self.skipTest('fastprobe A5 saved perceptions not present')
            with np.load(perception) as d:
                plan = d['trajectory'].copy()
            if packet.exists():
                with packet.open('rb') as f:
                    obs, info = pickle.load(f)
                speed, steer = float(info['ego_velo']), float(info['ego_steer'])
            else:
                speed, steer = 0.0, 0.0  # step 0 packet is not persisted by the harness
            t0 = time.perf_counter()
            out = tc.optimize(plan, None, info_at(speed, steer), vehicle=VEHICLE)
            solve_s = time.perf_counter() - t0
            self.assertIsNotNone(out['selected'], f'step {step}: {out.get("reason")}')
            decision = folder / 'control_decision.json'
            mpc_action = None
            if decision.exists():
                mpc_action = _hugsim_expand(json.loads(decision.read_text()))
            rows.append(dict(step=step, solve_seconds=solve_s,
                             tracking_first_action=out['selected']['controls'][0],
                             mpc_step_record=mpc_action))
            self.assertLess(solve_s, 5.0)
        report = dict(scope='offline replay of saved fastprobe A5 perceptions; diagnostic uncertified; '
                            'tracking path has no obstacle optimization (not safety-equivalent to MPC)',
                      rows=rows)
        out_path = HERE / 'tracking_replay_A5.json'
        out_path.write_text(json.dumps(report, indent=2))
        print('\nreplay ->', out_path)
        for r in rows:
            print('step', r['step'], 'solve', round(r['solve_seconds'], 3), 's',
                  'tracking', [round(v, 3) for v in r['tracking_first_action']])


class TestFullReexport(unittest.TestCase):
    """Regression: 2026-10-29 step-0 launch crash (missing polygon_check)."""

    def test_all_public_names_reexported(self):
        missing = [n for n in dir(tc._real)
                   if not n.startswith('_') and n != 'optimize' and not hasattr(tc, n)]
        self.assertEqual(missing, [])

    def test_consumers_imports_resolve(self):
        # Names imported lazily by harness-adjacent modules at call time.
        for name in ('polygon_check', 'body_vertices', 'asset_reference_vertices',
                     'initial_native_pose', 'reference_check', 'integrate_state'):
            self.assertTrue(callable(getattr(tc, name)), name)
        # optimize itself must be the tracking replacement, not the real MPC.
        self.assertIs(tc.optimize.__module__, 'tracking_common')


class TestPlanningGuardReexport(unittest.TestCase):
    """Regression: third launch failure (missing rectangle_separation)."""

    def test_all_public_names_reexported(self):
        missing = [n for n in dir(pgt._real)
                   if not n.startswith('_') and n not in ('POLICY', 'RecoveryController')
                   and not hasattr(pgt, n)]
        self.assertEqual(missing, [])

    def test_lazy_consumer_imports_resolve(self):
        self.assertTrue(callable(pgt.rectangle_separation))
        self.assertTrue(callable(pgt.ObstacleMemory))
        self.assertIn('NOT the evidence-tiered MPC', pgt.POLICY)
        self.assertTrue(issubclass(pgt.RecoveryController, pgt._real.RecoveryController))


if __name__ == '__main__':
    unittest.main(verbosity=2)

