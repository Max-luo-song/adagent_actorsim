"""Joint preflight must preserve body state and clear probe-only history."""

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
pass  # Path is provided by the portable boundary above
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'framework_backend'))
from native_edit import translate_native
from native_edit_probe import probe_native_edit, verify_translation
from runtime_contract import CAMERAS


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'ego_pose').mkdir()
        for frame in (0, 80):
            np.savetxt(self.root / f'ego_pose/{frame:03d}.txt', np.eye(4))
        model = SimpleNamespace(instances_trans=torch.zeros(90, 2, 3),
            instances_quats=torch.ones(90, 2, 4), smpl_qauts=torch.ones(90, 2, 23, 4))
        self.backend = SimpleNamespace(spec=SimpleNamespace(source=dict(
            inputs=str(self.root), start_timestep=0)), torch=torch, _last=None,
            trainer=SimpleNamespace(models={'SMPLNodes': model}),
            env=SimpleNamespace(models={'SMPLNodes': {'instances_trans': model.instances_trans}}),
            reset_count=0)
        def reset():
            self.backend._last = None
            self.backend.reset_count += 1
        self.backend.reset = reset
        self.edit = dict(schema='framework_native_edit_v1', operation='translate',
            group='SMPLNodes', instance=0, offset_world_m=[.5, 0, 0])

    def render(self, backend, pose, timestamp):
        moved = float(backend.trainer.models[self.edit['group']].instances_trans[80, 0, 0]) > 0
        cameras = {}
        for name in CAMERAS:
            rgb = np.zeros((4, 6, 3), dtype=np.uint8)
            mask = np.zeros((4, 6), dtype=np.uint8)
            rgb[1, 2 + int(moved)] = 100
            mask[1, 2 + int(moved)] = 255
            cameras[name] = dict(rgb=rgb, weather_dynamic_mask=mask,
                                  K=np.eye(3), camera_to_world=pose.copy())
        backend._last = (timestamp, pose)
        return dict(cameras=cameras)

    def test_edit_is_visible_body_preserved_and_probe_history_reset(self):
        report = probe_native_edit(self.backend, self.edit, self.root / 'probe', self.render)
        self.assertTrue(report['passed'])
        self.assertIsNone(self.backend._last)
        self.assertEqual(self.backend.reset_count, 2)
        self.assertTrue(torch.all(self.backend.trainer.models['SMPLNodes'].instances_trans[:, 0, 0] == .5))
        self.assertTrue(report['body_parameters_unchanged'])
        self.assertTrue((self.root / 'probe/CAM_FRONT_before_mask.png').exists())

    def test_body_corruption_prevents_joint_run(self):
        def corrupt(backend, edit):
            result = translate_native(backend, edit)
            backend.trainer.models['SMPLNodes'].smpl_qauts.add_(1)
            return result
        with patch('native_edit.translate_native', corrupt):
            with self.assertRaisesRegex(RuntimeError, 'native edit probe failed'):
                probe_native_edit(self.backend, self.edit, self.root / 'probe', self.render)
        report = _hugsim_expand(json.loads((self.root / 'probe/report.json').read_text()))
        self.assertFalse(report['body_parameters_unchanged'])
        self.assertIsNone(self.backend._last)

    def test_hidden_edit_not_accepted_as_visible_integration(self):
        first = self.render(self.backend, np.eye(4), 8.)
        self.backend.reset()
        def fixed_view(backend, pose, timestamp):
            return first
        with self.assertRaisesRegex(RuntimeError, 'native edit probe failed'):
            probe_native_edit(self.backend, self.edit, self.root / 'probe', fixed_view)

    def test_translation_rejects_extra_changes_to_other_instances(self):
        before = torch.zeros(3, 2, 3)
        after = before.clone()
        after[:, 0, 0] = .5
        self.assertTrue(verify_translation(torch, before, after, 0, [.5, 0, 0]))
        after[0, 1, 0] = .1
        self.assertFalse(verify_translation(torch, before, after, 0, [.5, 0, 0]))

    def test_probe_dispatches_through_unified_editor_once(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'unified'))
        from edit_api import SceneEditor, from_native_request
        session = SimpleNamespace(backend=self.backend, generation='probe-scene', revision=0, objects={})
        editor = SceneEditor(session, lambda value: None)
        command = from_native_request(self.edit)
        report = probe_native_edit(self.backend, self.edit, self.root / 'probe', self.render,
                                   apply_edit=lambda legacy: editor.edit(command))
        self.assertTrue(report['passed'])
        self.assertEqual(report['edit']['schema'], 'scene_edit_receipt_v1')
        self.assertEqual(session.revision, 1)
        self.assertEqual(len(editor.journal), 1)
        self.assertEqual(self.backend.reset_count, 3)

    def test_rigid_and_deformable_keep_orientation_and_network(self):
        for group in ('RigidNodes', 'DeformableNodes'):
            with self.subTest(group=group):
                model = SimpleNamespace(instances_trans=torch.zeros(90, 2, 3),
                    instances_quats=torch.ones(90, 2, 4))
                if group == 'DeformableNodes':
                    model.deform_network = torch.nn.Linear(3, 3)
                self.backend.trainer.models[group] = model
                self.backend.env.models[group] = {'instances_trans': model.instances_trans}
                self.edit['group'] = group
                report = probe_native_edit(self.backend, self.edit, self.root / group, self.render)
                self.assertTrue(report['passed'])
                self.assertTrue(report['other_groups_unchanged'])

    def test_deform_network_mutation_prevents_acceptance(self):
        model = self.backend.trainer.models.pop('SMPLNodes')
        model.deform_network = torch.nn.Linear(3, 3)
        self.backend.trainer.models['DeformableNodes'] = model
        self.backend.env.models['DeformableNodes'] = self.backend.env.models.pop('SMPLNodes')
        self.edit['group'] = 'DeformableNodes'
        def corrupt(edit):
            result = translate_native(self.backend, edit)
            with torch.no_grad(): model.deform_network.weight.add_(1)
            return result
        with self.assertRaisesRegex(RuntimeError, 'native edit probe failed'):
            probe_native_edit(self.backend, self.edit, self.root / 'corrupt', self.render, corrupt)
        report = _hugsim_expand(json.loads((self.root / 'corrupt/report.json').read_text()))
        self.assertFalse(report['deformation_network_unchanged'])

    def test_mutation_of_another_group_prevents_acceptance(self):
        other = SimpleNamespace(instances_trans=torch.zeros(90, 1, 3))
        self.backend.trainer.models['RigidNodes'] = other
        def corrupt(edit):
            result = translate_native(self.backend, edit)
            other.instances_trans.add_(1)
            return result
        with self.assertRaisesRegex(RuntimeError, 'native edit probe failed'):
            probe_native_edit(self.backend, self.edit, self.root / 'other', self.render, corrupt)
        report = _hugsim_expand(json.loads((self.root / 'other/report.json').read_text()))
        self.assertFalse(report['other_groups_unchanged'])


if __name__ == '__main__':
    unittest.main()
