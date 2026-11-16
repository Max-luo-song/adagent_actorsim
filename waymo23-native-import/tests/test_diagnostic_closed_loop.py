import sys,importlib.util
from pathlib import Path
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from edit_trust import assert_edit_evaluable,EditTrustError,diagnostic_closed_loop,diagnostic_closed_loop_enabled

class DiagnosticClosedLoopTests(unittest.TestCase):
    def test_metadata_cannot_enable_bypass_and_scope_resets_after_exception(self):
        data=dict(edited_scene=True,diagnostic_closed_loop=True,edit_trust=dict(edited=True,status='diagnostic_unverified'))
        with self.assertRaises(EditTrustError):assert_edit_evaluable(data)
        with self.assertRaises(RuntimeError):
            with diagnostic_closed_loop():
                self.assertTrue(diagnostic_closed_loop_enabled());assert_edit_evaluable(data)
                with self.assertRaises(EditTrustError):assert_edit_evaluable(data,expected_scene_digest='certificate-required')
                raise RuntimeError('simulate failed run')
        self.assertFalse(diagnostic_closed_loop_enabled())
        with self.assertRaises(EditTrustError):assert_edit_evaluable(data)
    def test_standalone_and_plugin_style_imports_share_only_scoped_flag(self):
        path=Path(__file__).resolve().parents[2]/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/edit_trust.py'
        spec=importlib.util.spec_from_file_location('test_plugin_edit_trust_alias',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.assertFalse(module.diagnostic_closed_loop_enabled())
        with diagnostic_closed_loop():self.assertTrue(module.diagnostic_closed_loop_enabled())
        self.assertFalse(module.diagnostic_closed_loop_enabled())
    def test_named_profile_and_planning_are_global_and_formal_defaults_unchanged(self):
        from shared_rules import shared_rules
        rules=shared_rules('current','calibration_geometry_v1')
        self.assertEqual(rules['model_input']['resize_mode'],'letterbox')
        self.assertEqual(rules['model_input']['camera_embedding_policy'],'geometry_only')
        self.assertEqual(rules['model_input']['planning_bev'],'current')
        self.assertEqual(shared_rules()['model_input']['planning_bev'],'history')
