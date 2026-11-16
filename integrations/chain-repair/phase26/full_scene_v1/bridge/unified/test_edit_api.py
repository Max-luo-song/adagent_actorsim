import copy
from types import SimpleNamespace
import unittest

import torch
from edit_api import SceneEditor, UnsupportedEditError, from_native_request, read_command


def command(group='SMPLNodes', offset=None):
    return dict(schema='scene_edit_v1', mode='diagnostic_preview', operation='translate',
        object_id=f'native:{group}:0', offset_world_m=offset or [.5, 0, 0])


class EditorTests(unittest.TestCase):
    def setUp(self):
        models = {name: SimpleNamespace(instances_trans=torch.zeros(6, 2, 3))
                  for name in ('RigidNodes', 'SMPLNodes', 'DeformableNodes')}
        self.backend = SimpleNamespace(torch=torch, _last=None,
            trainer=SimpleNamespace(models=models),
            env=SimpleNamespace(models={n: {'instances_trans': m.instances_trans}
                                        for n, m in models.items()}))
        self.backend.reset_count = 0
        def reset():
            self.backend.reset_count += 1
            self.backend._last = None
        self.backend.reset = reset
        self.session = SimpleNamespace(backend=self.backend, generation='scene-A', revision=0, objects={})
        self.editor = SceneEditor(self.session, lambda value: None)

    def test_all_native_actor_groups_share_command_semantics(self):
        for group in ('RigidNodes', 'SMPLNodes', 'DeformableNodes'):
            result = self.editor.edit(command(group))
            self.assertTrue(torch.all(self.backend.trainer.models[group].instances_trans[:, 0, 0] == .5))
            self.assertTrue(torch.all(self.backend.trainer.models[group].instances_trans[:, 1] == 0))
            self.assertEqual(result['edit_revision'], self.session.revision)
        self.assertEqual(self.session.revision, 3)
        self.assertEqual(self.backend.reset_count, 3)
        self.assertEqual(len(self.editor.objects()), 6)

    def test_failed_reset_rolls_back_geometry_and_revision(self):
        def failed():
            raise RuntimeError('weather history reset failed')
        self.backend.reset = failed
        with self.assertRaises(RuntimeError):
            self.editor.edit(command())
        self.assertEqual(self.session.revision, 0)
        self.assertFalse(self.editor.journal)
        self.assertFalse(self.editor.originals)
        self.assertTrue(torch.all(self.backend.trainer.models['SMPLNodes'].instances_trans == 0))

    def test_cumulative_translation_limit_and_restore_original(self):
        self.editor.edit(command(offset=[1.5, 0, 0]))
        self.editor.edit(command(offset=[.5, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'cumulative'):
            self.editor.edit(command(offset=[.1, 0, 0]))
        self.assertEqual(self.session.revision, 2)
        self.editor.restore_native()
        self.assertTrue(torch.all(self.backend.trainer.models['SMPLNodes'].instances_trans == 0))

    def test_stale_revision_generation_and_runtime_edits_rejected(self):
        for value in (dict(command(), expected_revision=1),
                      dict(command(), expected_generation='scene-B')):
            with self.assertRaises(ValueError): self.editor.edit(value)
        self.editor.started = True
        self.backend.reset()  # A low-level reset cannot bypass session phase.
        with self.assertRaises(RuntimeError): self.editor.edit(command())
        self.assertEqual(self.session.revision, 0)

    def test_stale_controller_rejected_after_scene_change(self):
        self.session.generation = 'scene-B'
        with self.assertRaises(RuntimeError): self.editor.objects()
        with self.assertRaises(RuntimeError): self.editor.edit(command())

    def test_proxy_identity_mismatch_is_atomic(self):
        self.backend.env.models['SMPLNodes']['instances_trans'] = torch.zeros(6, 2, 3)
        with self.assertRaises(RuntimeError): self.editor.edit(command())
        self.assertEqual(self.session.revision, 0)
        self.assertTrue(torch.all(self.backend.trainer.models['SMPLNodes'].instances_trans == 0))

    def test_command_and_receipt_mutation_cannot_rewrite_journal(self):
        value = command()
        result = self.editor.edit(value)
        value['offset_world_m'][0] = 1.7
        result['command']['offset_world_m'][0] = 1.8
        self.assertEqual(self.editor.journal[0]['command']['offset_world_m'][0], .5)
        objects = self.editor.objects()
        objects[0]['translation_world_m'][0] = 99
        self.assertNotEqual(self.editor.objects()[0]['translation_world_m'][0], 99)

    def test_framework_managed_insert_rejected_before_asset_loading(self):
        with self.assertRaises(UnsupportedEditError):
            self.editor.edit(dict(schema='scene_edit_v1', mode='diagnostic_preview',
                operation='insert', object_id='managed:car', asset='/not-loaded',
                asset_calibration='/not-loaded', asset_to_world=[]))
        self.assertEqual(self.session.revision, 0)

    def test_legacy_managed_adapter_keeps_one_revision_and_normalizes_id(self):
        session = SimpleNamespace(backend=SimpleNamespace(), generation='legacy', revision=0, objects={})
        seen = []
        def legacy(value):
            seen.append(copy.deepcopy(value))
            session.objects[value['object_id']] = {}
            session.revision += 1
            return dict(edit_revision=session.revision)
        editor = SceneEditor(session, legacy)
        result = editor.edit(dict(schema='scene_edit_v1', mode='diagnostic_preview',
            operation='insert', object_id='managed:car', asset='/asset', asset_calibration='/calib',
            asset_to_world=[]))
        self.assertEqual(seen[0]['object_id'], 'car')
        self.assertNotIn('schema', seen[0])
        self.assertEqual(result['edit_revision'], 1)
        self.assertEqual(editor.objects()[0]['object_id'], 'managed:car')
        with self.assertRaises(UnsupportedEditError): editor.edit(command())

    def test_invalid_commands_do_not_mutate(self):
        for field, value in (('offset_world_m', [True, 0, 0]), ('offset_world_m', [float('nan'),0,0]),
                             ('expected_revision', True), ('object_id', 'native:SMPLNodes:-1'),
                             ('object_id', 'native:SMPLNodes:99'), ('mode', 'formal')):
            with self.assertRaises(ValueError): self.editor.edit(dict(command(), **{field:value}))
        with self.assertRaises(ValueError): read_command(dict(command(), typo=True))
        with self.assertRaises(UnsupportedEditError): read_command(dict(command(), operation='scale'))
        self.assertEqual(self.session.revision, 0)

    def test_legacy_native_request_conversion(self):
        c = from_native_request(dict(schema='framework_native_edit_v1', operation='translate',
            group='SMPLNodes', instance=0, offset_world_m=[.5,0,0]))
        self.assertEqual(c, command())

    def test_framework_with_managed_adapter_delegates_asset_commands(self):
        self.backend.supports_managed_assets=True
        seen=[]
        def delegate(value):
            seen.append(value)
            self.session.objects[value['object_id']]={}
            self.session.revision+=1
            return dict(edit_revision=self.session.revision)
        editor=SceneEditor(self.session,delegate)
        self.assertIn('insert',editor.capabilities()['managed_operations'])
        editor.edit(dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='insert',
            object_id='managed:car',asset='/asset',asset_calibration='/calib',asset_to_world=[]))
        self.assertEqual(seen[0]['object_id'],'car')
        self.assertEqual(self.session.revision,1)

    def test_native_edit_clears_history_without_deleting_managed_assets(self):
        retained={'car':object()}
        self.backend.reset=lambda:retained.clear()
        self.backend.clear_observation_history=lambda:None
        self.editor.edit(command())
        self.assertIn('car',retained)


if __name__ == '__main__':
    unittest.main()
