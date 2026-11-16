from dataclasses import replace
from pathlib import Path
import sys
import unittest
import tempfile
import subprocess
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"waymo172_closed_loop_v1")]
from runtime_configuration import load_vehicle_config, scene_camera_mapping, preflight_source
from closed172_common import body_vertices, rollout, emergency_stop_action, integrate_state, optimize


class RuntimeConfigurationTests(unittest.TestCase):
    def test_reliability_config_loads_in_isolated_resident_worker(self):
        adapter = Path(__file__).resolve().parents[1]
        code = ("import sys; sys.path.insert(0, "+repr(str(adapter))+"); "
                "from runtime_configuration import configured_reliability; "
                "config = configured_reliability(); "
                "assert config.ego_half_width_m > 0")
        result = subprocess.run([sys.executable, '-I', '-c', code],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_preflight_rejects_corrupt_calibration_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'intrinsics').mkdir()
            (root/'extrinsics').mkdir()
            for cid in (0, 1, 2):
                np.savetxt(root/'intrinsics'/f'{cid}.txt', [800., 800., 400., 300.])
                np.savetxt(root/'extrinsics'/f'{cid}.txt', np.eye(4))
            source = dict(inputs=str(root), camera_ids=[0, 1, 2], pose_fps_assumption=10.)
            self.assertEqual(len(preflight_source(source)['cameras']), 3)
            invalid = np.eye(4)
            invalid[0, 0] = 2.
            np.savetxt(root/'extrinsics'/'1.txt', invalid)
            with self.assertRaisesRegex(ValueError, 'calibration'):
                preflight_source(source)

    def test_unseen_camera_ids_are_read_from_metadata(self):
        source = dict(scene_adapter=dict(camera_names={"8": "CAM_FRONT", "4": "CAM_FRONT_LEFT", "6": "CAM_FRONT_RIGHT"}))
        self.assertEqual(scene_camera_mapping(source)[8], "CAM_FRONT")
        for metadata in (dict(coordinate_system="unknown"),
                         dict(camera_names={"0": "CAM_FRONT", "1": "CAM_FRONT"}),
                         dict(threshold=.1)):
            with self.assertRaises(ValueError):
                scene_camera_mapping(dict(scene_adapter=metadata))

    def test_vehicle_profile_controls_footprint_and_dynamics(self):
        vehicle = replace(load_vehicle_config(), half_width_m=1.2, half_length_m=2.5,
                          wheelbase_m=3.6, dt=.05, emergency_braking_mps2=1.)
        body = body_vertices(np.eye(4), vehicle)
        np.testing.assert_allclose(np.ptp(body, axis=0), [2.4, 0., 5.])
        controls = np.tile([0., .05], (6, 1))
        states, _ = rollout(controls, 2., 0., [0, 0], vehicle.dt, vehicle.wheelbase_m)
        self.assertEqual(len(states), 61)
        reference = np.array([0., 0., 0., 2., 0.])
        for _ in range(60):
            reference = integrate_state(reference, controls[0], vehicle.dt, vehicle.wheelbase_m)
        np.testing.assert_allclose(states[-1], reference)
        for _ in range(50):
            action = emergency_stop_action(reference[3], vehicle.dt, vehicle.emergency_braking_mps2)
            reference = integrate_state(reference, [action['acc'], action['steer_rate']], vehicle.dt, vehicle.wheelbase_m)
            self.assertGreaterEqual(reference[3], -1e-12)
        self.assertAlmostEqual(reference[3], 0.)

    def test_optimizer_uses_configured_acceleration_and_rate_bounds(self):
        vehicle = replace(load_vehicle_config(), min_acc_mps2=-1., max_acc_mps2=.7,
                          max_steer_rate_degrees=5., emergency_braking_mps2=1.)
        occupancy = np.zeros((1, 5, 200, 200))
        occupancy[0, :, 190, 190] = 1
        info = dict(model_lidar_frame=dict(virtual_to_ego=np.eye(4)), ego_velo=1., ego_steer=0.)
        bounds_seen = []
        def solver(fun, seed, **kwargs):
            bounds_seen.append(kwargs['bounds'])
            self.assertTrue(np.isfinite(fun(seed)))
            self.assertTrue(np.isfinite(kwargs['constraints'][0]['fun'](seed)).all())
            return SimpleNamespace(x=seed, success=False, message='test solver')
        with patch('closed172_common.minimize', solver):
            result = optimize(np.zeros((6, 2)), occupancy, info, vehicle=vehicle)
        self.assertIsNotNone(result['selected'])
        self.assertGreaterEqual(result['selected']['minimum_constraint'], -1e-5)
        self.assertTrue(bounds_seen)
        self.assertEqual(bounds_seen[0][0], (-1., .7))
        np.testing.assert_allclose(bounds_seen[0][1], np.deg2rad([-5., 5.]))

    def test_invalid_vehicle_limits_fail_early(self):
        for values in (dict(wheelbase_m=0), dict(dt=.03), dict(max_steer_degrees=90),
                       dict(emergency_braking_mps2=3), dict(half_width_m=float('nan'))):
            with self.assertRaises(ValueError):
                replace(load_vehicle_config(), **values)
