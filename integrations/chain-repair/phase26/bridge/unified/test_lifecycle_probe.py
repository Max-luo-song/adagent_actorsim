"""CPU integration regression using real session/editor and causal buffer."""

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

BRIDGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BRIDGE / 'weather_runtime'))
sys.path.insert(0, str(BRIDGE / 'weather_sequence'))
from causal_buffer import CausalWeatherBuffer
from lifecycle_probe import probe_lifecycle
from scene_edit_session import UnifiedSceneSession, saf
from test_edit_api import command


class Backend:
    def __init__(self, spec):
        self.spec, self.torch, self._last, self.closed = spec, torch, None, False
        models = {group: SimpleNamespace(instances_trans=torch.zeros(90, 1, 3))
                  for group in ('RigidNodes', 'DeformableNodes')}
        self.trainer = SimpleNamespace(models=models)
        self.env = SimpleNamespace(models={g: {'instances_trans': m.instances_trans} for g,m in models.items()})
        self.managed_assets = SimpleNamespace(assets={})
        self.weather_records, self.epoch = [], 0
        self.buffer = CausalWeatherBuffer()
        self.last_weather_reset_verified = False
        self.bad_restoration = False
        self.bias = 0

    def clear_observation_history(self):
        self._last = None
        self.epoch += 1
        self.buffer = CausalWeatherBuffer()
        self.last_weather_reset_verified = True

    def reset(self):
        self.managed_assets.assets.clear()
        self.clear_observation_history()
        if self.bad_restoration: self.bias += 1

    def close(self): self.closed = True

    def clear_render(self, pose, timestamp):
        value = int(sum(float(m.instances_trans[80,0].sum()) for m in self.trainer.models.values()) * 20) + self.bias
        self._last = (timestamp, pose)
        return dict(ego_state={}, cameras={name: dict(rgb=np.full((4,6,3), value, dtype=np.uint8),
            weather_dynamic_mask=np.full((4,6), value, dtype=np.uint8), depth=np.ones((4,6),dtype=np.float32),
            K=np.eye(3), camera_to_world=pose.copy(), source_camera_id=i)
            for i,name in enumerate(self.spec.camera_order)})

    def render(self, pose, timestamp):
        raw = self.clear_render(pose, timestamp)
        _, audit = self.buffer.process(int(round(timestamp*10)), timestamp, str(self.epoch),
            np.zeros((3,4,6,3), dtype=np.uint8), np.zeros((3,4,6), dtype=np.uint8), lambda images, masks: images)
        self.weather_records.append(dict(audit=audit))
        return raw


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'ego_pose').mkdir()
        for frame in (0,80): np.savetxt(self.root/f'ego_pose/{frame:03d}.txt', np.eye(4))
        def spec(manifest):
            return SimpleNamespace(source=dict(scene=552 if manifest=='first' else 553,
                inputs=str(self.root), start_timestep=0, checkpoint=manifest), fps=10., frames=90,
                camera_order=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'), summary=lambda: {})
        reader = patch.object(saf.SceneSpec, 'read', side_effect=spec)
        reader.start(); self.addCleanup(reader.stop)
        self.backends = []
        def factory(spec):
            backend = Backend(spec); self.backends.append(backend); return backend
        self.session = UnifiedSceneSession(factory)
        self.addCleanup(self.session.close)
        self.commands = [command('RigidNodes'), command('DeformableNodes')]

    def run_probe(self, switch=None):
        return probe_lifecycle(self.session, 'first', self.commands, self.root/'probe',
            lambda backend, pose, timestamp: backend.clear_render(pose,timestamp), switch)

    def test_same_session_two_episodes_then_reload(self):
        report = self.run_probe()
        self.assertTrue(report['passed'])
        self.assertFalse(report['different_scene_verified'])
        self.assertFalse(report['uniad_persistent_worker_reset_verified'])
        self.assertTrue(self.backends[0].closed)
        self.assertEqual(len(self.backends), 2)
        self.assertEqual(len(self.backends[0].weather_records), 4)

    def test_distinct_scene_and_old_commands_rejected(self):
        report = self.run_probe('second')
        self.assertTrue(report['different_scene_verified'])
        self.assertEqual(report['reload_source']['scene'], 553)
        self.assertTrue(report['checks']['reload_old_command_rejected'])

    def test_same_source_is_not_certified_as_different_scene(self):
        with self.assertRaisesRegex(RuntimeError, 'scene reload lifecycle probe failed'):
            self.run_probe('first')
        report = _hugsim_expand(json.loads((self.root/'probe/report.json').read_text()))
        self.assertFalse(report['passed'])
        self.assertFalse(report['checks']['distinct_scene_source'])

    def test_failed_pixel_restoration_retains_failed_evidence(self):
        factory = self.session.factory
        def bad_factory(spec):
            backend = factory(spec); backend.bad_restoration = True; return backend
        self.session.factory = bad_factory
        with self.assertRaisesRegex(RuntimeError, 'episode lifecycle probe failed'):
            self.run_probe()
        report = _hugsim_expand(json.loads((self.root/'probe/report.json').read_text()))
        self.assertFalse(report['passed'])
        self.assertFalse(report['checks']['episode0_restored_pixels'])


if __name__ == '__main__': unittest.main()
