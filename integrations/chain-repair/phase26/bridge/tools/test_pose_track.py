"""CPU tests for pose_track: interpolation, rotation, time bounds, reset
reproducibility, and the edit-vs-animation revision contract."""
import math
import unittest

import numpy as np

from pose_track import PoseTrack, SceneEditState, slerp


def pose(x=0., y=0., z=0., yaw=0.):
    P = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    P[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    P[:3, 3] = [x, y, z]
    return P


class TrackTest(unittest.TestCase):
    def setUp(self):
        self.poses = [pose(float(i), 0., 0., 0.1*i) for i in range(5)]
        self.track = PoseTrack.create('car', 10., self.poses, [True]*5, edit_revision=1)

    def test_exact_frame_sampling(self):
        for i in range(5):
            np.testing.assert_allclose(self.track.sample(i/10.), self.poses[i], atol=1e-12)

    def test_linear_translation_midpoint(self):
        P = self.track.sample(0.15)
        np.testing.assert_allclose(P[:3, 3], [1.5, 0., 0.], atol=1e-9)

    def test_slerp_rotation_midpoint(self):
        P = self.track.sample(0.05)  # halfway yaw 0 -> 0.1
        yaw = math.atan2(P[1, 0], P[0, 0])
        self.assertAlmostEqual(yaw, 0.05, places=9)
        np.testing.assert_allclose(P[:3, :3].T@P[:3, :3], np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(P[:3, :3])), 1., places=9)

    def test_slerp_sign_symmetry(self):
        q = np.array([1., 0., 0., 0.])
        q2 = -q  # same rotation, opposite hemisphere
        out = slerp(q, q2, 0.5)
        np.testing.assert_allclose(np.abs(out), [1., 0., 0., 0.], atol=1e-12)

    def test_time_bounds(self):
        with self.assertRaises(ValueError):
            self.track.sample(-0.01)
        with self.assertRaises(ValueError):
            self.track.sample(0.401)  # beyond last frame (0.4)
        with self.assertRaises(ValueError):
            self.track.sample(float('nan'))

    def test_visibility_gating(self):
        t = PoseTrack.create('car', 10., self.poses, [True, False, False, True, True], 1)
        self.assertIsNone(t.sample(0.15))       # both endpoints invisible
        self.assertIsNone(t.sample(0.1))        # exact invisible frame
        np.testing.assert_allclose(t.sample(0.0), self.poses[0])

    def test_serialization_roundtrip(self):
        back = PoseTrack.from_dict(self.track.to_dict())
        self.assertEqual(back.edit_revision, 1)
        for a, b in zip(back.poses, self.poses):
            np.testing.assert_allclose(a, b, atol=0.)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PoseTrack.create('car', 10., self.poses, [True]*4, 1)   # length mismatch
        with self.assertRaises(ValueError):
            PoseTrack.create('car', 0., self.poses, [True]*5, 1)
        with self.assertRaises(ValueError):
            PoseTrack.create('car', 10., [pose()+0.01], [True], 1)  # non-rigid


class EditStateTest(unittest.TestCase):
    def setUp(self):
        self.state = SceneEditState(generation='g0')
        self.poses = [pose(float(i)) for i in range(5)]

    def test_animation_does_not_bump_revision(self):
        self.state.insert_track(PoseTrack.create('car', 10., self.poses, [True]*5, 1))
        token = self.state.model_cache_token
        self.state.animate('car', 0.1)
        self.state.animate('car', 0.2)
        self.assertEqual(self.state.model_cache_token, token)
        self.assertEqual(self.state.revision, 1)

    def test_edit_events_bump_revision(self):
        self.state.insert_track(PoseTrack.create('car', 10., self.poses, [True]*5, 1))
        t1 = self.state.model_cache_token
        self.state.replace_track(PoseTrack.create('car', 10., self.poses[::-1], [True]*5, 2))
        t2 = self.state.model_cache_token
        self.assertNotEqual(t1, t2)
        self.state.remove('car')
        self.assertNotEqual(t2, self.state.model_cache_token)
        self.assertEqual(self.state.revision, 3)

    def test_revision_ordering_enforced(self):
        self.state.insert_track(PoseTrack.create('car', 10., self.poses, [True]*5, 1))
        with self.assertRaises(ValueError):
            self.state.replace_track(PoseTrack.create('car', 10., self.poses, [True]*5, 1))
        with self.assertRaises(ValueError):
            self.state.insert_track(PoseTrack.create('car', 10., self.poses, [True]*5, 2))

    def test_remove_missing(self):
        with self.assertRaises(KeyError):
            self.state.remove('nope')

    def test_reset_reproducibility(self):
        # reset = new generation at revision 0; same inputs -> same state
        def build():
            s = SceneEditState(generation='g1')
            s.insert_track(PoseTrack.create('car', 10., self.poses, [True]*5, 1))
            return s
        a, b = build(), build()
        self.assertEqual(a.model_cache_token, b.model_cache_token)
        np.testing.assert_allclose(a.animate('car', 0.15), b.animate('car', 0.15), atol=0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
