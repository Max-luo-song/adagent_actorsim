"""Exercise the actual resident-worker hooks without importing GPU libraries."""
import ast
import copy
import inspect
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_profiles import resolve_model_profile


class ZeroTrainingBevSplitTests(unittest.TestCase):
    def hooks(self, policy):
        tree = ast.parse((ROOT/'waymo172_closed_loop_v1/closed172_worker.py').read_text())
        serve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'serve')
        hooks = [node for node in serve.body if isinstance(node, ast.FunctionDef) and node.name in ('get_bevs', 'planning')]
        calls = []
        def original_bevs(imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
            calls.append(dict(metas=copy.deepcopy(img_metas), prev_bev=prev_bev, prev_img=prev_img, prev_img_metas=prev_img_metas))
            return ('history', 'history_pos') if prev_bev is not None else ('fresh', 'fresh_pos')
        def original_forward(bev_embed, bev_pos, motion_query):
            return bev_embed, bev_pos, motion_query
        env = dict(inspect=inspect, copy=copy, original_bevs=original_bevs,
                   original_forward=original_forward, PLANNING_BEV=policy, current={})
        exec(compile(ast.Module(body=hooks, type_ignores=[]), '<actual-worker-hooks>', 'exec'), env)
        return env, calls

    def test_current_planning_preserves_history_perception_and_metadata(self):
        env, calls = self.hooks('current')
        bus = np.arange(18, dtype=float)
        metadata = [dict(can_bus=bus.copy(), scene_token='unseen-scene')]
        self.assertEqual(env['get_bevs']('images', metadata, prev_bev='cached'), ('history', 'history_pos'))
        self.assertEqual(env['planning']('history', 'history_pos', 'history-motion'), ('fresh', 'fresh_pos', 'history-motion'))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['prev_bev'], 'cached')
        self.assertIsNone(calls[1]['prev_bev'])
        self.assertIsNone(calls[1]['prev_img'])
        self.assertIsNone(calls[1]['prev_img_metas'])
        np.testing.assert_array_equal(metadata[0]['can_bus'], bus)
        expected = bus.copy(); expected[:3] = 0; expected[-1] = 0
        np.testing.assert_array_equal(calls[1]['metas'][0]['can_bus'], expected)
        self.assertEqual(env['current']['planning_calls'], 1)

    def test_history_reference_does_not_compute_extra_bev(self):
        env, calls = self.hooks('history')
        self.assertEqual(env['get_bevs']('images', [dict(can_bus=np.zeros(18))], prev_bev='cached'), ('history', 'history_pos'))
        self.assertEqual(env['planning']('history', 'history_pos', 'motion'), ('history', 'history_pos', 'motion'))
        self.assertEqual(len(calls), 1)

    def test_learned_adapter_and_training_requests_fail_before_model_loading(self):
        for field, value in [('train_native_regression', True), ('train_native_regression', False),
                             ('regression_adapter_checkpoint', '/tmp/learned.pth'), ('training_epochs', 20)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Zero-new-training'):
                resolve_model_profile({field: value})

    def test_global_ab_rules_change_only_planning_policy(self):
        from shared_rules import shared_rules, validate_case_rules
        from edit_trust import digest
        history, current = shared_rules('history'), shared_rules('current')
        current['protocol']['planning_bev'] = 'history'
        current['model_input']['planning_bev'] = 'history'
        self.assertEqual(history, current)
        rules = shared_rules('current')
        for scene in ('172', '552', 'unseen'):
            case = dict(source_report=scene, protocol_hash=digest(rules['protocol']), rules_hash=digest(rules))
            self.assertEqual(validate_case_rules(case, planning_bev='current')['rules_hash'], digest(rules))
            with self.assertRaises(ValueError):
                validate_case_rules(case)
        with self.assertRaises(ValueError):
            shared_rules('scene_172_offset')
