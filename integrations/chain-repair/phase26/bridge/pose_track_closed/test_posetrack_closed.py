"""CPU tests for the scene_api pose_track shim (mock backend; no GPU)."""

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

import sys
import types
import unittest
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
for p in (str(HERE), str(HERE.parents[0] / 'pose_track'),
          str(ROOT / 'integrations/sr-hugsim')):
    if p not in sys.path:
        sys.path.insert(0, p)

import scene_api_posetrack as sap


def make_instance():
    """SceneSession instance without running the real __init__."""
    inst = sap.SceneSession.__new__(sap.SceneSession)
    inst.revision = 1
    inst.generation = 3
    inst.objects = {}
    inst.backend = None
    return inst


class TestReexportCompleteness(unittest.TestCase):
    def test_all_public_names_reexported(self):
        missing = [n for n in dir(sap._real)
                   if not n.startswith('_') and n != 'SceneSession' and not hasattr(sap, n)]
        self.assertEqual(missing, [])

    def test_session_is_subclass(self):
        self.assertTrue(issubclass(sap.SceneSession, sap._real.SceneSession))
        self.assertIsNot(sap.SceneSession, sap._real.SceneSession)


class TestEditTrajectoryPop(unittest.TestCase):
    def setUp(self):
        self.recorded = {}
        real_edit = sap._real.SceneSession.edit
        def fake_edit(self, request):
            self.recorded_edit = request
            return {'ok': True}
        self.patcher = patch = __import__('unittest.mock', fromlist=['mock'])
        sap._real.SceneSession.edit = fake_edit
        self.addCleanup(setattr, sap._real.SceneSession, 'edit', real_edit)

    def test_plain_insert_passthrough(self):
        inst = make_instance()
        req = dict(operation='insert', object_id='x')
        inst.edit(req)
        self.assertEqual(inst.recorded_edit, req)  # untouched
        self.assertFalse(hasattr(inst, '_pose_track_adapter'))

    def test_trajectory_field_popped_before_delegate(self):
        inst = make_instance()
        anchor = np.eye(4)
        poses = np.stack([anchor, anchor + np.array([[0, 0, 0, 1.2]]).T * 0], axis=0)
        poses = np.stack([anchor, anchor.copy()], axis=0)
        poses[1][0, 3] = 1.2
        track = dict(times_s=[8.0, 8.6], asset_to_world_4x4=poses.tolist(), bounds='strict')
        req = dict(operation='insert', object_id='car', pose_track_trajectory=track)
        # GPU applier is not available on CPU: inject a fake gpu_applier module.
        applied = []
        fake_ga = types.ModuleType('gpu_applier')
        fake_ga.make_backend_pose_applier = lambda backend: (lambda oid, pose: applied.append((oid, np.asarray(pose))))
        applied = []
        sys.modules['gpu_applier'] = fake_ga
        self.addCleanup(sys.modules.pop, 'gpu_applier')
        inst.backend = object()  # non-None so trajectory registration proceeds
        inst.edit(req)
        self.assertNotIn('pose_track_trajectory', inst.recorded_edit)
        adapter = inst._pose_track_adapter
        self.assertIn('car', adapter.registry.object_ids())
        # trajectory endpoints preserved
        np.testing.assert_allclose(adapter.registry.render_pose('car'), poses[0])

    def test_trajectory_shape_validation(self):
        inst = make_instance()
        inst.backend = object()
        fake_ga = types.ModuleType('gpu_applier')
        fake_ga.make_backend_pose_applier = lambda backend: (lambda oid, pose: None)
        sys.modules['gpu_applier'] = fake_ga
        self.addCleanup(sys.modules.pop, 'gpu_applier')
        bad = dict(times_s=[8.0], asset_to_world_4x4=[np.eye(4).tolist(), np.eye(4).tolist()])
        with self.assertRaises(ValueError):
            inst.edit(dict(operation='insert', object_id='c', pose_track_trajectory=bad))


class TestRenderAdvance(unittest.TestCase):
    def test_render_advances_before_delegate(self):
        calls = []
        real_render = sap._real.SceneSession.render
        sap._real.SceneSession.render = lambda self, pose, ts, **kw: calls.append(('render', float(ts))) or {'packet': True}
        self.addCleanup(setattr, sap._real.SceneSession, 'render', real_render)
        inst = make_instance()
        adapter = types.SimpleNamespace(
            registry=types.SimpleNamespace(object_ids=lambda: ['car']),
            advance_pose_track=lambda t: calls.append(('advance', t)))
        inst._pose_track_adapter = adapter
        out = inst.render(np.eye(4), 8.1)
        self.assertEqual(calls, [('advance', 8.1), ('render', 8.1)])
        self.assertEqual(out, {'packet': True})

    def test_render_without_track_is_passthrough(self):
        calls = []
        real_render = sap._real.SceneSession.render
        sap._real.SceneSession.render = lambda self, pose, ts, **kw: calls.append(('render', float(ts))) or {}
        self.addCleanup(setattr, sap._real.SceneSession, 'render', real_render)
        inst = make_instance()
        inst.render(np.eye(4), 8.0)
        self.assertEqual(calls, [('render', 8.0)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
