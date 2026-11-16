"""CPU fixture tests for the pose_track probe logic (no GPU, no torch)."""

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
import sys
import unittest
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
JOB = HERE.parents[3]
sys.path[:0] = [str(HERE), str(JOB / "chain_repair/phase26/bridge/pose_track")]

from pose_track import PoseTrajectory
import pose_track_probe as P


def _packet(t, frame, rev, gen, state_seed):
    rng = np.random.default_rng(abs(hash((frame, state_seed))) % (2**32))
    cams = {}
    for name in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"):
        cams[name] = {"rgb": rng.integers(0, 255, (4, 6, 3), dtype=np.uint8)}
    return {"cameras": cams, "timestamp_s": t, "source_frame": frame,
            "history_token": gen, "edit_revision": rev,
            "model_cache_token": f"{gen}:{rev}",
            "ego_to_world": np.eye(4)}


class MockAsset:
    def __init__(self, pose):
        self.source_pose = np.asarray(pose, dtype=float).copy()
        T = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        self.native_pose = T @ self.source_pose


class MockBackend:
    def __init__(self):
        self.assets = {}


class MockSession:
    def __init__(self):
        self.backend = MockBackend()
        self.revision = 0
        self.generation = "fixture-gen"
        self.t = 0.0
        self.objects = {}

    def observe(self):
        return self._packet()

    def step(self, action):
        self.t += 0.1
        return self._packet()

    def _packet(self):
        seed = hash(tuple(sorted(
            (k, tuple(np.asarray(v.source_pose).ravel().round(9)))
            for k, v in self.backend.assets.items())))
        return _packet(self.t, int(round(self.t * 10)), self.revision, self.generation, seed)

    def edit(self, request):
        if request["operation"] == "insert":
            if request["object_id"] in self.objects:
                raise ValueError("Object ID already exists")
            self.objects[request["object_id"]] = dict(request)
            self.backend.assets[request["object_id"]] = MockAsset(request["asset_to_world"])
        elif request["operation"] == "remove":
            del self.objects[request["object_id"]]
            del self.backend.assets[request["object_id"]]
        self.revision += 1
        return {"object_id": request["object_id"], "edit_revision": self.revision}

    def render(self, ego, ts):
        seed = hash(tuple(sorted(
            (k, tuple(np.asarray(v.source_pose).ravel().round(9)))
            for k, v in self.backend.assets.items())))
        return _packet(ts, int(round(ts * 10)), self.revision, self.generation, seed)

    def close(self):
        pass


class MockApplier:
    """Recording applier mirroring gpu_applier's contract (updates backend in place)."""
    def __init__(self, backend, broken=False):
        self.backend = backend
        self.broken = broken
        self.calls = []

    def __call__(self, object_id, pose):
        self.calls.append((object_id, np.asarray(pose).copy()))
        if not self.broken:
            self.backend.assets[object_id] = MockAsset(pose)


def make_plan():
    plan = _hugsim_expand(json.loads((HERE / "plan.json").read_text()))
    plan["sources"] = {}  # fixtures do not pin GPU sources
    return plan


class ProbeFlowTests(unittest.TestCase):
    def run_b(self, broken_applier=False):
        session = MockSession()
        applier = MockApplier(session.backend, broken=broken_applier)
        return P.run_arm(session, "B", make_plan(), pose_applier=applier), applier

    def test_b_arm_completes_and_evaluates_pass(self):
        ra = P.run_arm(MockSession(), "A", make_plan())
        rb, _ = self.run_b()
        ev = P.evaluate({"A": ra, "B": rb})
        self.assertTrue(ev["passed"], msg=json.dumps(ev["criteria"], indent=1))

    def test_broken_applier_fails_readback(self):
        ra = P.run_arm(MockSession(), "A", make_plan())
        rb, _ = self.run_b(broken_applier=True)
        ev = P.evaluate({"A": ra, "B": rb})
        self.assertFalse(ev["criteria"]["readback_exact"])
        self.assertFalse(ev["passed"])

    def test_applier_called_each_advance_with_expected_pose(self):
        rb, applier = self.run_b()
        traj = P.build_trajectory(make_plan())
        self.assertEqual(len(applier.calls), 6)  # steps 2..7
        for (oid, pose), readback in zip(applier.calls, rb["readbacks"]):
            expected = traj.evaluate(readback["t"])
            np.testing.assert_allclose(pose, expected, atol=0, rtol=0)

    def test_out_of_range_and_duplicate_rejected(self):
        rb, _ = self.run_b()
        self.assertTrue(rb["out_of_range_advance"]["rejected"])
        self.assertTrue(rb["duplicate_insert"]["rejected"])

    def test_revision_bumps_only_on_structural_edits(self):
        rb, _ = self.run_b()
        revs = [o["meta"]["edit_revision"] for o in rb["observations"]]
        self.assertEqual(revs[:2], [0, 0])
        self.assertTrue(all(r == 1 for r in revs[2:]))

    def test_trajectory_strict_bounds(self):
        traj = P.build_trajectory(make_plan())
        with self.assertRaises(ValueError):
            traj.evaluate(0.9)
        with self.assertRaises(ValueError):
            traj.evaluate(0.1)
        mid = traj.evaluate(0.45)
        self.assertAlmostEqual(mid[0, 3], 18.82 + 0.25, places=12)

    def test_plan_contract_fields(self):
        spec = make_plan()["trajectory"]
        self.assertEqual(spec["unit"], "meter")
        self.assertIn("scene_session_world", spec["coordinate_frame"])
        self.assertEqual(spec["time_window_s"], [0.2, 0.7])
        self.assertEqual(spec["bounds"], "strict")

    def test_non_claims_recorded(self):
        text = " ".join(make_plan()["non_claims"])
        self.assertIn("UniAD history continuity NOT validated", text)


if __name__ == "__main__":
    unittest.main()
