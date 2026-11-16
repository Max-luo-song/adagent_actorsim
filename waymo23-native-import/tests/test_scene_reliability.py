"""CPU regression tests for evidence and its effect on actual MPC constraints."""
from pathlib import Path
import sys
import ast
import copy
from types import SimpleNamespace
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"))
sys.path.insert(0, str(BASE/"waymo23-native-import/waymo172_closed_loop_v1"))
from scene_reliability import (SceneReliability, ReliabilityConfig,
                               augment_occupancy, limit_acceleration,
                               pose_from_row_rotation)
from closed172_common import optimize


def proposals(x=0., y=10., score=.22):
    # One real-sized proposal above a flat weak-query background.
    boxes = np.tile([25., 30., 0., 1.8, 4., 1.5, 0., 0., 0.], (21, 1))
    boxes[0, :2] = [x, y]
    scores = np.full(21, .02)
    scores[0] = score
    return boxes, scores, np.zeros(21, dtype=int)


PLAN = np.column_stack([np.zeros(6), np.arange(1, 7)*2.])


class ReliabilityTests(unittest.TestCase):
    def update(self, monitor, time=0., token="unseen", pose=None, **kw):
        boxes, scores, labels = proposals(**kw)
        return monitor.update(boxes, scores, labels, time, token,
                              pose=np.eye(4) if pose is None else pose, trajectory=PLAN)

    def test_low_score_reaches_guard_without_promoting_score(self):
        report = self.update(SceneReliability())
        self.assertTrue(report["uncertain_path_conflicts"])
        self.assertEqual(report["obstacles"][0]["raw_score"], .22)
        self.assertFalse(report["raw_scores_modified"])
        self.assertEqual(report["obstacles"][0]["state"], "provisional_uncertain")
        self.assertEqual(report["speed_limit_mps"], 2.)

    def test_world_association_across_ego_rotation_and_translation(self):
        m = SceneReliability()
        world = np.array([0., 10., 0., 1.])
        for i in range(3):
            angle = i*.2
            c, s = np.cos(angle), np.sin(angle)
            pose = np.array([[c, -s, 0, i], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
            local = np.linalg.inv(pose) @ world
            report = self.update(m, time=i*.1, pose=pose, x=local[0], y=local[1])
        self.assertEqual(report["obstacles"][0]["state"], "persistent_uncertain")
        self.assertEqual(report["obstacles"][0]["observations"], 3)
        self.assertTrue(np.allclose(pose_from_row_rotation(pose[:3, :3].T, pose[:3, 3]), pose))

    def test_time_based_confirmation_at_different_rates(self):
        for dt in [.1, .5]:
            m = SceneReliability()
            for i in range(3):
                report = self.update(m, time=i*dt)
            self.assertEqual(report["obstacles"][0]["state"], "persistent_uncertain")

    def test_no_pose_does_not_manufacture_temporal_evidence(self):
        m = SceneReliability()
        b, s, l = proposals()
        for i in range(4):
            r = m.update(b, s, l, i*.1, "s", trajectory=PLAN)
        self.assertEqual(r["obstacles"][0]["observations"], 1)

    def test_scene_time_reset_and_association_outlier(self):
        m = SceneReliability()
        self.update(m)
        self.update(m, time=.1)
        r = self.update(m, time=.2, token="other")
        self.assertEqual(r["obstacles"][0]["observations"], 1)
        r = self.update(m, time=.2, token="other")
        self.assertEqual(r["reset_reason"], "non_increasing_time")
        r = self.update(m, time=.3, token="other", x=10.)
        self.assertEqual(r["obstacles"][0]["observations"], 1)
        r = self.update(m, time=5., token="other")
        self.assertEqual(r["reset_reason"], "time_gap")

    def test_flat_noise_and_off_path_do_not_request_braking(self):
        b, s, l = proposals(score=.02)
        r = SceneReliability().update(b, s, l, 0, "s", trajectory=PLAN)
        self.assertEqual(r["obstacles"], [])
        r = self.update(SceneReliability(), x=20.)
        self.assertEqual(r["action"], "unchanged")

    def test_single_frame_does_not_become_confirmed_by_duplicate_queries(self):
        b, s, l = proposals()
        b[1] = b[0]
        s[1] = .21
        r = SceneReliability().update(b, s, l, 0, "s", pose=np.eye(4), trajectory=PLAN)
        self.assertEqual(len(r["obstacles"]), 1)
        self.assertEqual(r["obstacles"][0]["observations"], 1)

    def test_bad_geometry_and_calibration_rejected(self):
        b, s, l = proposals()
        b[0, 3] = -1
        r = SceneReliability().update(b, s, l, 0, "s", trajectory=PLAN)
        self.assertEqual(r["obstacles"], [])
        with self.assertRaises(ValueError):
            self.update(SceneReliability(), pose=np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            ReliabilityConfig(candidate_floor=2.)

    def test_real_camera_projection_and_visibility(self):
        # Camera looks along LiDAR +y; vertical points project near mid-height.
        p = np.array([[100, 50, 0, 0], [0, 50, -100, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
        b, s, l = proposals()
        meta = dict(lidar2img=[p], img_shape=[[100, 100, 3]])
        m = SceneReliability()
        r = m.update(b, s, l, 0, "s", metadata=meta, trajectory=PLAN)
        self.assertTrue(r["obstacles"][0]["visible"])
        b[0, 1] = -10
        r = m.update(b, s, l, .1, "s", metadata=meta, trajectory=PLAN)
        self.assertEqual(r["obstacles"], [])

    def test_guard_occupancy_is_separate_and_marks_low_score_vehicle(self):
        r = self.update(SceneReliability())
        for dtype in [np.float32, np.int64, bool]:
            original = np.zeros((1, 5, 1, 200, 200), dtype=dtype)
            guarded = augment_occupancy(original, r)
            self.assertFalse(original.any())
            self.assertTrue(guarded[0, :, 0, 120, 100].all())

    def test_mpc_consumes_uncertainty_and_enforces_reachable_speed_limit(self):
        # Off-path weak proposal: no additional obstacle conflicts. A report cap
        # is supplied to isolate actual MPC speed constraint behavior.
        report = self.update(SceneReliability(), x=20.)
        report["speed_limit_mps"] = 2.
        occupancy = np.zeros((1, 5, 1, 200, 200), dtype=np.uint8)
        occupancy[0, :, 0, 190, 190] = 1
        info = dict(ego_velo=3., ego_steer=0.,
                    model_lidar_frame=dict(virtual_to_ego=np.eye(4).tolist()))
        result = optimize(PLAN, occupancy, info, maxiter=120, reliability=report)
        self.assertTrue(result["uncertainty_guard_enabled"])
        self.assertIsNotNone(result["selected"])
        states = np.asarray(result["selected"]["states"])
        envelope = np.maximum(2., 3.-2.*np.arange(len(states))*.1)
        self.assertTrue(np.all(states[:, 3] <= envelope+1e-5))
        self.assertAlmostEqual(limit_acceleration(1., 3., .1, report), -2.)
        self.assertEqual(limit_acceleration(1., 3., .1, {}), 1.)

    def test_actual_uniad_forward_exports_unfiltered_uncertainty(self):
        # Execute the real orchestration function with CPU heads. This catches
        # integration errors without requiring mmcv CUDA operators or weights.
        import torch
        source = BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_test")
        pop_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pop_elem_in_result")
        scope = dict(copy=copy, pose_from_row_rotation=pose_from_row_rotation)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[pop_fn, fn], type_ignores=[])), str(source), "exec"), scope)
        b, s, l = proposals()
        tracking = dict(bev_embed=torch.zeros(1), boxes_3d_det=b, scores_3d_det=s,
                        labels_3d_det=l, boxes_3d=np.empty((0, 9)),
                        scores_3d=np.empty(0), labels_3d=np.empty(0))
        dummy = SimpleNamespace(
            prev_frame_info=dict(scene_token=None, prev_bev=None), video_test_mode=True,
            with_seg_head=False, with_motion_head=False, with_occ_head=False, with_planning_head=True,
            simple_test_track=lambda *args: [tracking], upsample_bev_if_tiny=lambda v: v,
            planning_head=SimpleNamespace(forward_test=lambda *args: dict(sdc_traj=torch.tensor(PLAN[None]))),
            scene_reliability=SceneReliability(), track_base=SimpleNamespace(score_thresh=.4))
        metas = [[dict(scene_token="new-scene", can_bus=np.zeros(18))]]
        result = scope["forward_test"](dummy, img=torch.zeros(1), img_metas=metas,
                                      l2g_r_mat=np.eye(3), l2g_t=np.zeros(3), timestamp=0.)
        self.assertEqual(len(result[0]["scores_3d"]), 0)
        self.assertEqual(result[0]["scene_reliability"]["action"], "caution")
        np.testing.assert_array_equal(result[0]["scores_3d_det"], s)

if __name__ == "__main__":
    unittest.main()
