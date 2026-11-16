"""Data-driven start selection, independent ground fit and bounded GPU jobs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import ast
import copy
import inspect

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/"waymo23-native-import"))
from n1_cross_scene_audit import CrossSceneProtocol, source_geometry, fit_observed_ground
from build_source_sequence import recorded_kinematics, training_lidar_mount
from gpu_test_service import build_command
from edit_trust import digest


class CrossSceneProtocolTests(unittest.TestCase):
    def test_paired_comparison_dispatch_remains_typed_and_workspace_bounded(self):
        command = build_command(dict(action='compare_perception_pair', args=dict(pair=__file__)), BASE/'outputs/test-result')
        self.assertTrue(command[1].endswith('paired_edit_diagnostic.py'))
        for args in (dict(pair='/etc/passwd'), dict(pair=__file__, command='extra')):
            with self.assertRaises(ValueError):
                build_command(dict(action='compare_perception_pair', args=args), BASE/'outputs/test-result')

    def test_start_selection_is_invariant_to_global_origin_and_heading(self):
        for heading, translation in [(0., [0, 0, 0]), (.7, [30000, -50000, 12])]:
            with tempfile.TemporaryDirectory() as temporary:
                inputs = Path(temporary)
                (inputs/"ego_pose").mkdir()
                c, s = np.cos(heading), np.sin(heading)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
                for i in range(60):
                    p = np.eye(4)
                    p[:3, :3] = rotation
                    p[:3, 3] = rotation@np.array([i*.25, 0, 0])+translation
                    np.savetxt(inputs/"ego_pose"/f"{i:03d}.txt", p)
                geometry = source_geometry(dict(inputs=str(inputs), frames=60), CrossSceneProtocol())
                self.assertEqual(geometry["frame"], 0)
                np.testing.assert_allclose(geometry["anchor"][:2], [12, 0], atol=1e-8)
                self.assertAlmostEqual(geometry["speed_mps"], 2.5)

    def test_plane_fit_rejects_outliers_without_gaussian_road_geometry(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(-3, 3, (300, 2))
        z = .04*xy[:, 0]-.02*xy[:, 1]+1.2+rng.normal(0, .005, 300)
        points = np.column_stack([xy, z])
        points[:40, 2] += 2.
        result = fit_observed_ground(points, .1)
        a, b, height = result["plane"]
        self.assertAlmostEqual(a, .04, delta=.005)
        self.assertAlmostEqual(b, -.02, delta=.005)
        self.assertLess(result["rms_m"], .02)
        self.assertGreater(result["inliers"], 240)

    def test_recorded_metadata_uses_common_kinematic_rule(self):
        relative = np.tile(np.eye(4), (60, 1, 1))
        relative[:, 0, 3] = np.arange(60)*.25
        speed, steer, command = recorded_kinematics(relative, 20, 10., 2.7)
        self.assertAlmostEqual(speed, 2.5)
        self.assertAlmostEqual(steer, 0.)
        self.assertEqual(command, 2)
        mount = training_lidar_mount(BASE/"code/HUGSIM/configs/sim/nuscenes_camera.yaml")
        np.testing.assert_allclose(mount[:3, 3], [0., .985793, 1.84019], atol=1e-5)

    def test_gpu_service_only_dispatches_bounded_allowlisted_tests(self):
        source = str(BASE/"outputs/waymo552_three_setup_20261018_043131/source_report.json")
        command = build_command(dict(action="build_source_sequence", args=dict(source_report=source, start_frame=60, frames=3)), BASE/"outputs/test-result")
        self.assertIn("--start-frame", command)
        self.assertTrue(command[1].endswith("build_source_sequence.py"))
        for request in [dict(action="shell", args=dict(cmd="echo unauthorized")),
                        dict(action="build_source_sequence", args=dict(source_report=source, start_frame=0, frames=1000000)),
                        dict(action="render_scene", args=dict(source_report="/etc/passwd")),
                        dict(action="render_scene", args=dict(source_report=source, command="unexpected"))]:
            with self.assertRaises(ValueError):
                build_command(request, BASE/"outputs/test-result")

    def test_actual_worker_uses_history_without_recomputing_fresh_bev(self):
        self._worker_bev_policy("history", 1, "history")

    def test_current_bev_is_explicit_ablation_and_does_not_mutate_history_metadata(self):
        self._worker_bev_policy("current", 2, "fresh")

    def test_generic_closed_loop_rejects_unlocked_protocol_and_unverified_insert(self):
        from dataclasses import asdict
        from run_n1_closed_loop import run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root/"case.json"
            case.write_text(json.dumps(dict(protocol_hash="changed")))
            with self.assertRaisesRegex(ValueError, "locked shared"):
                run(case, root/"result")
            case.write_text(json.dumps(dict(protocol_hash=digest(asdict(CrossSceneProtocol())),
                source_report=str(BASE/"outputs/waymo172_three_setup_20261016_132050_295391/source_report.json"),
                asset_report=str(BASE/"outputs/waymo172_car_sequence_20261016_154023/asset_report.json"))))
            with self.assertRaisesRegex(ValueError, "independently revalidated"):
                run(case, root/"result")
            self.assertFalse((root/"result").exists())

    def _worker_bev_policy(self, mode, expected_calls, expected_bev):
        path = BASE/"waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py"
        tree = ast.parse(path.read_text())
        serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
        functions = [n for n in serve.body if isinstance(n, ast.FunctionDef) and n.name in ("get_bevs", "planning")]
        calls = []
        def bevs(img, img_metas, prev_bev=None, prev_img=None, prev_img_metas=None):
            calls.append((prev_bev, img_metas[0]["can_bus"].copy()))
            return ("history" if prev_bev is not None else "fresh"), "pos"
        def forward(bev_embed, bev_pos):
            return bev_embed
        scope = dict(inspect=inspect, copy=copy, original_bevs=bevs, original_forward=forward,
                     PLANNING_BEV=mode, current={})
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), scope)
        metadata = [dict(can_bus=np.full(18, 3.))]
        self.assertEqual(scope["get_bevs"](None, metadata, prev_bev=object())[0], "history")
        self.assertEqual(scope["planning"]("history", "pos"), expected_bev)
        self.assertEqual(len(calls), expected_calls)
        np.testing.assert_array_equal(metadata[0]["can_bus"], np.full(18, 3.))
        if mode == "current":
            np.testing.assert_array_equal(calls[-1][1][:3], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
