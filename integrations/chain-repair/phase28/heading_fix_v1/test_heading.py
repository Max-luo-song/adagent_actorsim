"""CPU regressions for sign, wrapping, native-frame adapter and v3 step40."""

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

import ast
import json
import math
pass  # Path is provided by the portable boundary above
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from heading_fix_v1.heading import (compute_heading_error, mpc_heading_error,
                                    spatial_tracking_cost, wrap_to_pi)
from heading_fix_v1 import closed172_common as fixed_mpc


def production_spatial_cost(source, fixture, **extra):
    """Evaluate the actual nested optimizer function on the saved fixture.

    Extract its AST to supply the real fixed input without running SLSQP or
    changing any scene state. This catches a helper that was never wired in.
    """
    tree = ast.parse(source.read_text())
    optimize = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'optimize')
    fn = next(n for n in optimize.body if isinstance(n, ast.FunctionDef)
              and n.name == 'spatial_tracking')
    scope = dict(np=np, spatial_guide=np.asarray(fixture['path']),
                 substeps=round(.5/fixture['dt']),
                 control_interval_s=fixture['control_interval_s'],
                 target_speed=fixture['target_speed'],
                 vehicle=SimpleNamespace(wheelbase_m=fixture['wheelbase']), **extra)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    return scope['spatial_tracking']


class HeadingRegression(unittest.TestCase):
    def test_heading_error_sign(self):
        self.assertGreater(compute_heading_error((10, 3), 0), 0)
        self.assertLess(compute_heading_error((10, -3), 0), 0)
        self.assertEqual(compute_heading_error((10, 0), 0), 0)

    def test_nonzero_ego_heading(self):
        path = (math.cos(math.radians(30)), math.sin(math.radians(30)))
        self.assertAlmostEqual(compute_heading_error(path, math.radians(45)),
                               math.radians(-15))
        self.assertAlmostEqual(compute_heading_error((1, 0), math.radians(-45)),
                               math.radians(45))

    def test_heading_wrap_across_pi(self):
        for target, ego, expected in [(-179, 179, 2), (179, -179, -2)]:
            with self.subTest(target=target):
                theta = math.radians(target)
                err = compute_heading_error((math.cos(theta), math.sin(theta)),
                                            math.radians(ego))
                self.assertAlmostEqual(err, math.radians(expected))
        self.assertEqual(wrap_to_pi(math.pi), -math.pi)
        self.assertEqual(compute_heading_error((-1, 0), 0), -math.pi)

    def test_joint_rotation_invariance(self):
        target, ego = .35, -.7
        expected = target - ego
        for rotation in [-2.9, -1., 0., 2.9]:
            direction = (math.cos(target+rotation), math.sin(target+rotation))
            self.assertAlmostEqual(compute_heading_error(direction, ego+rotation), expected)

    def test_batch_heading_errors(self):
        vectors = np.array([[10, 3], [10, -3], [10, 0]])
        errors = compute_heading_error(vectors, np.zeros(3))
        np.testing.assert_allclose(errors, [math.atan(.3), -math.atan(.3), 0], atol=1e-15)

    def test_invalid_or_undefined_heading_rejected(self):
        for vector in [(0, 0), (math.nan, 1), (math.inf, 1), (1,), (1, 2, 3)]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                compute_heading_error(vector, 0)
        with self.assertRaises(ValueError):
            compute_heading_error((1, 1), math.nan)
        with self.assertRaises(ValueError):
            compute_heading_error([[1, 1], [0, 0]], 0)

    def test_native_coordinate_adapter(self):
        self.assertAlmostEqual(mpc_heading_error((3, 10), .2),
                               compute_heading_error((10, 3), .2))
        self.assertGreater(mpc_heading_error((3, 10), 0), 0)
        self.assertLess(mpc_heading_error((-3, 10), 0), 0)

    def test_heading_agrees_with_actual_mpc_dynamics(self):
        for steer in [-.2, 0., .2]:
            first = fixed_mpc.integrate_state([0, 0, 0, 1, steer], [0, 0])
            second = fixed_mpc.integrate_state(first, [0, 0])
            displacement = second[:2] - first[:2]
            self.assertAlmostEqual(mpc_heading_error(displacement, first[2]), 0.)
            if steer:
                self.assertGreater(displacement[0]*steer, 0.)

    def test_cost_prefers_recovery(self):
        fixture = _hugsim_expand(json.loads((HERE/'fixtures/step40.json').read_text()))
        frozen = HERE.parent.parent/'phase11/closed172_common.py'
        legacy = production_spatial_cost(frozen, fixture)
        fixed = production_spatial_cost(HERE/'closed172_common.py', fixture,
                                       spatial_tracking_cost=fixed_mpc.spatial_tracking_cost)
        costs = {}
        for name, row in fixture['candidates'].items():
            self.assertTrue(row['feasible'])
            states = np.asarray(row['states'])
            # The remaining original objective terms stay exactly as recorded.
            costs[name] = row['objective'] - legacy(states) + fixed(states)
        keep = 'lane_change_right'  # seed names are not final directions
        recover = 'reachable_right_stop'
        self.assertLess(fixture['candidates'][keep]['controls'][0][1], 0)
        self.assertGreater(fixture['candidates'][recover]['controls'][0][1], 0)
        self.assertGreater(fixture['candidates'][recover]['objective'],
                           fixture['candidates'][keep]['objective'])
        self.assertLess(costs[recover], costs[keep])
        self.assertAlmostEqual(costs[recover], 123.20282634214155, places=7)
        self.assertAlmostEqual(costs[keep], 151.7953818286913, places=7)

    def test_heading_cost_does_not_always_prefer_right(self):
        path = np.array([[0., 0.], [-10., 10.]])  # genuinely leftward path
        aligned = np.array([[0, 0, -math.pi/4, 1, 0], [-.1, .1, -math.pi/4, 1, 0]])
        mirrored = aligned.copy()
        mirrored[:, 2] = math.pi/4
        cost = lambda s: spatial_tracking_cost(s, path, 1, 1., 1., 2.7)
        self.assertLess(cost(aligned), cost(mirrored))


if __name__ == '__main__':
    unittest.main(verbosity=2)
