from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scene_edit_session import UnifiedSceneSession, saf
from test_edit_api import command


class Backend:
    def __init__(self, spec):
        self.spec = spec
        self.torch = torch
        trans = torch.zeros(4, 1, 3)
        self.trainer = SimpleNamespace(models={'SMPLNodes': SimpleNamespace(instances_trans=trans)})
        self.env = SimpleNamespace(models={'SMPLNodes': {'instances_trans': trans}})
        self._last = None
        self.closed = False
        self.history = []
    def reset(self): self._last = None; self.history.clear()
    def close(self): self.closed = True
    def render(self, pose, timestamp):
        self._last = (timestamp, pose)
        self.history.append(timestamp)
        camera = dict(rgb=np.zeros((4,6,3), dtype=np.uint8), depth=np.ones((4,6),dtype=np.float32),
                      K=np.eye(3), camera_to_world=pose.copy(), source_camera_id=0)
        return dict(cameras={name: camera.copy() for name in self.spec.camera_order}, ego_state={})


class SessionTests(unittest.TestCase):
    def setUp(self):
        spec = SimpleNamespace(source={'scene':552}, fps=10., frames=4,
            camera_order=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'),
            summary=lambda: dict(backend='legacy', full_sr_trainer=False))
        self.reader = patch.object(saf.SceneSpec, 'read', return_value=spec)
        self.reader.start()
        self.addCleanup(self.reader.stop)
        self.session = UnifiedSceneSession(Backend)
        self.session.load_scene('fake')
        self.addCleanup(self.session.close)

    def test_packet_metadata_revision_and_reset_restoration(self):
        s = self.session
        generation = s.generation
        old_editor = s.editor
        s.edit(command())
        packet = s.render(np.eye(4), 0.)
        self.assertTrue(packet['edit_trust']['edited'])
        self.assertTrue(packet['full_sr_trainer'])
        self.assertEqual(packet['backend'], 'framework_renderer')
        self.assertEqual(packet['edit_revision'], 1)
        with self.assertRaises(RuntimeError): s.edit(command())
        result = s.reset()
        self.assertEqual(result['edit_revision'], 0)
        self.assertNotEqual(s.generation, generation)
        self.assertIsNone(s.backend._last)  # reset does not render or repopulate history
        self.assertFalse(s.backend.history)
        self.assertTrue(torch.all(s.backend.trainer.models['SMPLNodes'].instances_trans == 0))
        with self.assertRaises(RuntimeError): old_editor.edit(command())
        s.edit(command())  # new pre-run phase works

    def test_scene_switch_closes_backend_and_invalidates_editor(self):
        old_backend, old_editor = self.session.backend, self.session.editor
        self.session.edit(command())
        self.session.load_scene('another')
        self.assertTrue(old_backend.closed)
        self.assertFalse(self.session.edit_journal())
        with self.assertRaises(RuntimeError): old_editor.objects()

    def test_journal_return_value_is_independent(self):
        self.session.edit(command())
        journal = self.session.edit_journal()
        journal[0]['command']['offset_world_m'][0] = 7
        self.assertEqual(self.session.edit_journal()[0]['command']['offset_world_m'][0], .5)


if __name__ == '__main__': unittest.main()
