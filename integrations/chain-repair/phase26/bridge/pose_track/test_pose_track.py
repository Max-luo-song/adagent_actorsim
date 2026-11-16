"""CPU tests for pose_track: interpolation, rotation, bounds, reset
reproduction, render/collision pose identity, and cache-token handling."""

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
import unittest
pass  # Path is provided by the portable boundary above
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_track import PoseTrajectory, PoseTrackRegistry, _slerp, _rot_to_quat, _quat_to_rot
from session_adapter import PoseTrackSession


def pose(x=0., y=0., z=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    m[:3, 3] = [x, y, z]
    return m


class TrajectoryTests(unittest.TestCase):
    def test_endpoints_exact(self):
        tr = PoseTrajectory([1., 3.], [pose(0., yaw=0.), pose(4., yaw=1.0)])
        np.testing.assert_allclose(tr.evaluate(1.), pose(0., yaw=0.), atol=1e-12)
        np.testing.assert_allclose(tr.evaluate(3.), pose(4., yaw=1.0), atol=1e-12)

    def test_midpoint_lerp_and_slerp(self):
        tr = PoseTrajectory([0., 2.], [pose(0., 0.), pose(10., 6., yaw=np.pi / 2)])
        mid = tr.evaluate(1.)
        np.testing.assert_allclose(mid[:3, 3], [5., 3., 0.], atol=1e-12)
        np.testing.assert_allclose(mid[:3, :3], pose(yaw=np.pi / 4)[:3, :3], atol=1e-12)

    def test_rotation_stays_orthonormal_right_handed(self):
        tr = PoseTrajectory([0., 1., 2.5],
                            [pose(yaw=0.), pose(yaw=2.0), pose(yaw=-1.0)])
        for t in np.linspace(0., 2.5, 26):
            r = tr.evaluate(t)[:3, :3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(r), 1.0, places=9)

    def test_slerp_quaternion_roundtrip(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            rot = pose(yaw=float(rng.uniform(-np.pi, np.pi)))[:3, :3]
            q = _rot_to_quat(rot)
            np.testing.assert_allclose(_quat_to_rot(q), rot, atol=1e-12)
        # slerp endpoints
        q0 = _rot_to_quat(pose(yaw=0.)[:3, :3]); q1 = _rot_to_quat(pose(yaw=1.3)[:3, :3])
        np.testing.assert_allclose(_slerp(q0, q1, 0.), q0, atol=1e-12)
        np.testing.assert_allclose(_slerp(q0, q1, 1.), q1, atol=1e-12)

    def test_bounds_strict_raises(self):
        tr = PoseTrajectory([1., 2.], [pose(), pose(1.)])
        with self.assertRaises(ValueError):
            tr.evaluate(0.5)
        with self.assertRaises(ValueError):
            tr.evaluate(2.5)

    def test_bounds_hold_clamps(self):
        tr = PoseTrajectory([1., 2.], [pose(0.), pose(4.)], bounds='hold')
        np.testing.assert_allclose(tr.evaluate(-5.)[:3, 3], [0., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(tr.evaluate(99.)[:3, 3], [4., 0., 0.], atol=1e-12)

    def test_static_single_keyframe(self):
        tr = PoseTrajectory([8.], [pose(1., 2., yaw=.5)], bounds='hold')
        np.testing.assert_allclose(tr.evaluate(8.0), pose(1., 2., yaw=.5), atol=1e-12)
        np.testing.assert_allclose(tr.evaluate(123.0), pose(1., 2., yaw=.5), atol=1e-12)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):  # non-monotonic
            PoseTrajectory([0., 0.], [pose(), pose(1.)])
        with self.assertRaises(ValueError):  # decreasing
            PoseTrajectory([2., 1.], [pose(), pose(1.)])
        with self.assertRaises(ValueError):  # NaN time
            PoseTrajectory([0., np.nan], [pose(), pose(1.)])
        with self.assertRaises(ValueError):  # shape mismatch
            PoseTrajectory([0., 1.], [pose()])
        with self.assertRaises(ValueError):  # empty
            PoseTrajectory([], np.zeros((0, 4, 4)))
        with self.assertRaises(ValueError):  # non-rigid rotation
            bad = np.eye(4); bad[0, 0] = 2.
            PoseTrajectory([0.], [bad])
        with self.assertRaises(ValueError):  # left-handed
            bad = np.eye(4); bad[0, 0] = -1.
            PoseTrajectory([0.], [bad])


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.reg = PoseTrackRegistry()
        self.tr = PoseTrajectory([0., 10.], [pose(0.), pose(20.)])

    def test_edit_events_bump_revision_and_token(self):
        t0 = self.reg.cache_token
        self.reg.insert('car', self.tr)
        self.assertEqual(self.reg.revision, 1)
        self.assertNotEqual(self.reg.cache_token, t0)
        self.reg.advance(5.)
        tok = self.reg.cache_token
        self.reg.set_trajectory('car', self.tr)
        self.assertEqual(self.reg.revision, 2)
        self.assertNotEqual(self.reg.cache_token, tok)
        tok = self.reg.cache_token
        self.reg.remove('car')
        self.assertEqual(self.reg.revision, 3)
        self.assertNotEqual(self.reg.cache_token, tok)

    def test_advance_never_changes_token(self):
        self.reg.insert('car', self.tr)
        tok = self.reg.cache_token
        for k in range(1, 11):
            self.reg.advance(float(k))
            self.assertEqual(self.reg.cache_token, tok)
            self.assertEqual(self.reg.revision, 1)

    def test_advance_updates_pose_and_time(self):
        self.reg.insert('car', self.tr)
        self.reg.advance(2.5)
        np.testing.assert_allclose(self.reg.render_pose('car')[:3, 3], [5., 0., 0.], atol=1e-12)
        self.assertEqual(self.reg.sim_time, 2.5)

    def test_time_backwards_rejected(self):
        self.reg.insert('car', self.tr)
        self.reg.advance(5.)
        with self.assertRaises(ValueError):
            self.reg.advance(4.)

    def test_reset_new_generation_clears_state(self):
        self.reg.insert('car', self.tr)
        self.reg.advance(3.)
        tok = self.reg.cache_token
        self.reg.reset()
        self.assertEqual(self.reg.revision, 0)
        self.assertEqual(self.reg.object_ids(), ())
        self.assertIsNone(self.reg.sim_time)
        self.assertNotEqual(self.reg.cache_token, tok)

    def test_reset_reproduction_bit_identical(self):
        def build():
            reg = PoseTrackRegistry()
            reg.insert('car', PoseTrajectory([0., 4.], [pose(0., yaw=.1), pose(8., yaw=1.1)]))
            reg.advance(0.0)
            reg.advance(1.3)
            reg.advance(2.7)
            return reg.render_pose('car')
        np.testing.assert_array_equal(build(), build())

    def test_render_and_collision_pose_are_the_same_array(self):
        self.reg.insert('car', self.tr)
        self.reg.advance(1.)
        self.assertIs(self.reg.render_pose('car'), self.reg.collision_pose('car'))

    def test_unknown_objects_rejected(self):
        with self.assertRaises(KeyError):
            self.reg.remove('ghost')
        with self.assertRaises(KeyError):
            self.reg.set_trajectory('ghost', self.tr)
        self.reg.insert('car', self.tr)
        with self.assertRaises(KeyError):
            self.reg.insert('car', self.tr)

    def test_strict_trajectory_outside_sim_time_rejected_on_insert(self):
        self.reg.advance(50.)
        with self.assertRaises(ValueError):
            self.reg.insert('car', self.tr)  # window [0,10] covers 50? yes; use short one
        short = PoseTrajectory([0., 1.], [pose(), pose(1.)])
        with self.assertRaises(ValueError):
            self.reg.insert('bike', short)


class MockSession:
    """SceneSession-shaped mock: edit bumps revision, render irrelevant."""
    def __init__(self):
        self.revision = 0
        self.generation = 'mock-gen'
        self.edits = []
        self.resets = 0
        self.objects = {}
    def edit(self, req):
        self.edits.append(req)
        self.revision += 1
        if req.get('op') == 'insert':
            self.objects['car'] = {'asset_to_world': pose(0.).tolist()}
        elif req.get('op') == 'remove':
            self.objects.pop('car', None)
        return {'edit_revision': self.revision}
    def reset(self):
        self.resets += 1
        self.revision = 0
        self.generation = 'mock-gen-%d' % self.resets


class SessionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.session = MockSession()
        self.applied = {}
        self.adapter = PoseTrackSession(self.session, lambda oid, p: self.applied.__setitem__(oid, p.copy()))
        self.tr = PoseTrajectory([0., 10.], [pose(0.), pose(20.)])

    def test_insert_via_edit_event_then_advance_without_revision_change(self):
        self.adapter.insert_tracked('car', self.tr, edit_request={'op': 'insert'})
        self.assertEqual(self.session.revision, 1)
        self.adapter.advance_pose_track(4.)
        self.assertEqual(self.session.revision, 1)  # no UniAD history reset
        np.testing.assert_allclose(self.applied['car'][:3, 3], [8., 0., 0.], atol=1e-12)

    def test_remove_via_edit_event(self):
        self.adapter.insert_tracked('car', self.tr, edit_request={'op': 'insert'})
        self.adapter.remove_tracked('car', edit_request={'op': 'remove'})
        self.assertEqual(self.session.revision, 2)
        self.assertEqual(self.adapter.registry.object_ids(), ())

    def test_advance_detects_revision_mutation(self):
        def bad_applier(oid, p):
            self.session.revision += 1  # simulate the bug being guarded against
        adapter = PoseTrackSession(self.session, bad_applier)
        adapter.insert_tracked('car', self.tr)
        with self.assertRaises(RuntimeError):
            adapter.advance_pose_track(1.)

    def test_reset_clears_both_sides(self):
        self.adapter.insert_tracked('car', self.tr, edit_request={'op': 'insert'})
        self.adapter.advance_pose_track(2.)
        self.adapter.reset()
        self.assertEqual(self.session.resets, 1)
        self.assertEqual(self.adapter.registry.object_ids(), ())
        self.assertEqual(self.session.revision, 0)

    def test_objects_metadata_synced_without_revision_change(self):
        self.adapter.insert_tracked('car', self.tr, edit_request={'op': 'insert'})
        np.testing.assert_allclose(self.session.objects['car']['asset_to_world'],
                                   pose(0.), atol=1e-12)
        self.adapter.advance_pose_track(4.)
        self.assertEqual(self.session.revision, 1)
        np.testing.assert_allclose(self.session.objects['car']['asset_to_world'],
                                   pose(8.), atol=1e-12)

    def test_packet_poses_match_collision_poses(self):
        self.adapter.insert_tracked('car', self.tr)
        self.adapter.advance_pose_track(3.)
        pkt = self.adapter.packet_poses()
        np.testing.assert_array_equal(pkt['car'], self.adapter.registry.collision_pose('car'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
