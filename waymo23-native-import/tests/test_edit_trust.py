"""Independent analytic CPU scenes exercise N1, including bad edits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"),
               str(BASE/"waymo23-native-import")]
from edit_trust import (EditTrustDomain, EditTrustError, assert_edit_evaluable,
                        digest, check_views, check_skin, EditPolicy)
from edit_trust_adapter import runtime_edit_metadata, require_edit_mutation, capture_local_views


def request(kind="insert", identity="edit-1", scene="baseline"):
    bounds = [[-1., -1., 0.], [1., 1., 2.]]
    surface = [[x, y, z] for x in [-1., 1.] for y in [-1., 1.] for z in [0., 2.]]
    surface += [[0., 0., 2.], [0., 0., 0.]]
    value = dict(edit_id=identity, kind=kind, scene_digest=scene,
                 geometry_skin=surface, support_points=[[x, y, 0.] for x in [-.6, .6] for y in [-.6, .6]])
    if kind in ("insert", "move"):
        value["new_bounds"] = bounds
    if kind != "insert":
        value["old_bounds"] = copy.deepcopy(bounds)
    return value


def ground_skin():
    x, y = np.meshgrid(np.arange(-2., 2.01, .1), np.arange(-2., 2.01, .1))
    return dict(ground_points=np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]),
                occupied_points=np.empty((0, 3)), source_id="independent-lidar", source_kind="sensor")


def analytic_view(req, stage="pre", revision=0, camera_id="left", camera_x=0.):
    h = w = 48
    k = np.array([[24., 0, 24], [0, 24., 24], [0, 0, 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [camera_x, 0., 8.]
    yy, xx = np.mgrid[:h, :w]
    ray = np.stack([(xx-24)/24, -(yy-24)/24, -np.ones_like(xx)], axis=-1)
    depth = np.full((h, w), 8., dtype=float)
    rgb = np.full((h, w, 3), .3, dtype=float)
    if req["kind"] in ("insert", "move"):
        bounds = np.asarray(req["new_bounds"])
        origins = pose[:3, 3]
        near = np.full((h, w), -np.inf)
        far = np.full((h, w), np.inf)
        for axis in range(3):
            d = ray[..., axis]
            nonzero = np.abs(d) > 1e-9
            t0 = np.divide(bounds[0, axis]-origins[axis], d, out=np.full_like(d, -np.inf), where=nonzero)
            t1 = np.divide(bounds[1, axis]-origins[axis], d, out=np.full_like(d, np.inf), where=nonzero)
            lower, upper = np.minimum(t0, t1), np.maximum(t0, t1)
            if not bounds[0, axis] <= origins[axis] <= bounds[1, axis]:
                lower[~nonzero], upper[~nonzero] = np.inf, -np.inf
            near, far = np.maximum(near, lower), np.minimum(far, upper)
        hit = (near > 0) & (far >= near) & (near < depth)
        depth[hit] = near[hit]
        rgb[hit] = .7
    return dict(request_hash=digest(req), stage=stage, revision=revision,
                camera_id=camera_id, render_id="render-"+camera_id+stage,
                reference_id="heldout-"+camera_id, reference_kind="heldout",
                K=k, camera_to_world=pose, depth=depth, reference_depth=depth.copy(),
                rgb=rgb, reference_rgb=rgb.copy(), alpha=np.ones((h, w)), roi=np.ones((h, w), bool))


def views(req, stage="pre", revision=0, scene=None):
    result = [analytic_view(req, stage, revision, "left", 0.),
              analytic_view(req, stage, revision, "right", .5)]
    for view in result:
        view["scene_digest"] = scene or req["scene_digest"]
    return result


def domain():
    return EditTrustDomain(np.full((24, 24), .95), np.full((24, 24), 3), [-6., -6.], .5,
                           "baseline-independent-observations", "baseline")


class EditTrustTests(unittest.TestCase):
    def test_sloped_asymmetric_ground_uses_plane_at_skin_point(self):
        contacts = np.array([[x, y, .3*x] for x in (-1., 1.) for y in (-1., 1.)])
        surface = np.vstack([contacts, contacts+[0, 0, 1.]])
        ground = np.array([[x+dx, y+dy, .3*(x+dx)]
                           for x, y, _ in contacts
                           for dx in (.35, .4, .45) for dy in (-.05, 0., .05)])
        req = dict(kind="insert", support_points=contacts, geometry_skin=surface,
                   new_bounds=[[-1.01, -1.01, -.31], [1.01, 1.01, 1.31]])
        skin = dict(ground_points=ground, source_kind="sensor", source_id="analytic-slope")
        check_skin(req, skin, EditPolicy())
        # Real penetration remains rejected even when contact patches are valid.
        req["geometry_skin"] = np.vstack([surface, [0.99, .99, 0.]])
        with self.assertRaisesRegex(EditTrustError, "penetrates"):
            check_skin(req, skin, EditPolicy())

    def test_valid_insert_quarantines_before_mutation_then_restores(self):
        d, req = domain(), request()
        original = d.confidence.copy()
        edit = d.begin(req, ground_skin(), views(req))
        self.assertLess(d.confidence.min(), d.policy.trust_threshold)
        with self.assertRaises(EditTrustError):
            d.assert_evaluable()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"trust.json"
            def mutation():
                on_disk = EditTrustDomain.load(path)
                with self.assertRaises(EditTrustError):
                    on_disk.assert_evaluable()
                return "edited-scene"
            d.apply(edit, mutation, "edited-scene", path)
            d.revalidate(edit, views(req, "post", d.revision, "edited-scene"), path)
            np.testing.assert_array_equal(d.confidence, original)
            assert_edit_evaluable(dict(edit_trust=EditTrustDomain.load(path).evaluation_metadata()))

    def test_unknown_or_unobserved_or_outside_region_rejected(self):
        for missing in ["low", "views", "outside"]:
            d, req = domain(), request()
            if missing == "low":
                d.confidence[12, 12] = .1
            elif missing == "views":
                d.observations[12, 12] = 1
            else:
                req["new_bounds"] = [[-10, -1, 0], [-8, 1, 2]]
            with self.assertRaises(EditTrustError):
                d.begin(req, ground_skin(), views(req))
            self.assertEqual(d.revision, 0)

    def test_floating_penetrating_and_collision_are_rejected(self):
        for error in ["float", "penetrate", "occupied", "missing_ground"]:
            req, skin = request(), ground_skin()
            if error in ("float", "penetrate"):
                delta = 1. if error == "float" else -.5
                req["support_points"] = (np.asarray(req["support_points"])+[0, 0, delta]).tolist()
                req["geometry_skin"] = (np.asarray(req["geometry_skin"])+[0, 0, delta]).tolist()
                req["new_bounds"] = (np.asarray(req["new_bounds"])+[0, 0, delta]).tolist()
            elif error == "occupied":
                skin["occupied_points"] = [[0., 0., 1.]]
            else:
                skin["ground_points"] = [[10, 10, 0], [11, 10, 0], [10, 11, 0]]
            with self.assertRaises(EditTrustError, msg=error):
                domain().begin(req, skin, views(req))

    def test_holes_ghosts_and_cross_view_breakage_are_rejected(self):
        for error in ["hole", "ghost", "depth", "cross_view"]:
            req = request()
            evidence = views(req)
            if error == "hole":
                evidence[1]["alpha"][20:30, 20:30] = 0
            elif error == "ghost":
                evidence[1]["rgb"][20:30, 20:30] = 0
            elif error == "depth":
                evidence[1]["depth"][20:30, 20:30] += 2
            else:
                # Individual references agree, but the two camera surfaces do not.
                evidence[1]["depth"] += 3
                evidence[1]["reference_depth"] += 3
            with self.assertRaises(EditTrustError, msg=error):
                domain().begin(req, ground_skin(), evidence)

    def test_incomplete_roi_duplicate_views_and_self_reference_rejected(self):
        for error in ["roi", "duplicate", "same_pose", "self", "stale"]:
            req = request()
            evidence = views(req)
            if error == "roi":
                evidence[0]["roi"][:] = False
                evidence[0]["roi"][:5, :5] = True
            elif error == "duplicate":
                evidence[1]["camera_id"] = evidence[0]["camera_id"]
            elif error == "same_pose":
                evidence[1]["camera_to_world"] = evidence[0]["camera_to_world"]
            elif error == "self":
                evidence[0]["reference_id"] = evidence[0]["render_id"]
            else:
                evidence[0]["revision"] = -1
            with self.assertRaises(EditTrustError, msg=error):
                domain().begin(req, ground_skin(), evidence)

    def test_individually_matching_references_cannot_bypass_cross_view_geometry(self):
        req = request("erase")
        evidence = views(req)
        evidence[1]["depth"] += 3.
        evidence[1]["reference_depth"] += 3.
        with self.assertRaisesRegex(EditTrustError, "multiview surface"):
            domain().begin(req, {}, evidence)

    def test_failed_postcheck_and_restart_keep_region_quarantined(self):
        d, req = domain(), request()
        edit = d.begin(req, ground_skin(), views(req))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"state.json"
            d.apply(edit, lambda: "edited", "edited", p)
            post = views(req, "post", d.revision, "edited")
            post[0]["alpha"][:] = 0
            with self.assertRaises(EditTrustError):
                d.revalidate(edit, post, p)
            restored = EditTrustDomain.load(p)
            with self.assertRaises(EditTrustError):
                restored.assert_evaluable()
            self.assertEqual(restored.edits[edit]["state"], "revalidation_failed")
            restored.revalidate(edit, views(req, "post", d.revision, "edited"), p)
            restored.assert_evaluable()

    def test_crash_readback_mismatch_and_tampering_block_evaluation(self):
        d, req = domain(), request()
        edit = d.begin(req, ground_skin(), views(req))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"state.json"
            with self.assertRaises(EditTrustError):
                d.apply(edit, lambda: "unexpected", "edited", p)
            with self.assertRaises(EditTrustError):
                EditTrustDomain.load(p).assert_evaluable()
            payload = json.loads(p.read_text())
            payload["edits"][edit]["state"] = "verified"
            p.write_text(json.dumps(payload))
            with self.assertRaises(EditTrustError):
                EditTrustDomain.load(p)

    def test_erase_repair_need_independent_revealed_surface_reference(self):
        for kind in ["erase", "repair"]:
            req = request(kind)
            d = domain()
            evidence = views(req)
            edit = d.begin(req, {}, evidence)
            self.assertEqual(d.edits[edit]["state"], "quarantined")
            evidence[0]["reference_kind"] = "independent_geometry"
            with self.assertRaises(EditTrustError):
                domain().begin(req, {}, evidence)

    def test_move_quarantines_both_original_and_destination(self):
        d, req = domain(), request("move")
        req["old_bounds"] = [[-4., -1., 0.], [-2., 1., 2.]]
        edit = d.begin(req, ground_skin(), views(req))
        cells = d.edits[edit]["cells"]
        self.assertIn((12, 5), cells)
        self.assertIn((12, 13), cells)

    def test_preview_and_forged_boolean_do_not_enter_evaluation(self):
        from types import SimpleNamespace
        env = SimpleNamespace()
        with self.assertRaises(EditTrustError):
            require_edit_mutation(env, "erase", dict(old_bounds=[[-1, -1, 0], [1, 1, 2]]))
        require_edit_mutation(env, "erase", dict(old_bounds=[[-1, -1, 0], [1, 1, 2]]), preview_only=True)
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable(dict(edit_trust=runtime_edit_metadata(env)))
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable(dict(edit_trust=dict(edited=True, verified=True)))
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable(dict(asset_motion="stationary at existing world placement"))

    def test_unverified_edit_blocks_high_score_model_and_unified_runner(self):
        from scene_reliability import SceneReliability
        from run_reliable_sequence import inspect_packet, SLOTS
        import pickle
        metadata = dict(edit_trust=dict(edited=True, verified=True))
        with self.assertRaises(EditTrustError):
            SceneReliability().update(np.array([[0., 10., 0., 2., 4., 1., 0.]]),
                np.array([.99]), np.array([0]), 0., "scene", metadata=metadata,
                trajectory=np.array([[0., 5.], [0., 10.]]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"packet.pkl"
            with p.open("wb") as stream:
                pickle.dump(({}, metadata), stream)
            with self.assertRaises(EditTrustError):
                inspect_packet(p, SLOTS)

    def test_certificate_binding_reuse_and_overlapping_edits_are_rejected(self):
        d, req = domain(), request()
        edit = d.begin(req, ground_skin(), views(req))
        with self.assertRaises(EditTrustError):
            d.begin(request(identity="edit-2"), ground_skin(), views(req))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"trust.json"
            d.apply(edit, lambda: "actual-scene", "actual-scene", p)
            with self.assertRaises(EditTrustError):
                d.revalidate(edit, views(req, "post", d.revision, "another-scene"), p)
            d.revalidate(edit, views(req, "post", d.revision, "actual-scene"), p)
            metadata = dict(edit_trust=d.evaluation_metadata())
            self.assertNotIn("confidence", metadata["edit_trust"]["trust_state"])
            assert_edit_evaluable(metadata, "actual-scene")
            with self.assertRaises(EditTrustError):
                assert_edit_evaluable(metadata, "another-scene")

    def test_unedited_trust_field_cannot_certify_an_inserted_vehicle(self):
        from edit_trust_adapter import load_evaluation_domain
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"baseline.json"
            domain().save(p)
            with self.assertRaises(EditTrustError):
                load_evaluation_domain(p, "baseline")
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable({}, "expected-edited-scene")
        with self.assertRaises(EditTrustError):
            assert_edit_evaluable(dict(edit_trust=dict(edited=False)), "expected-edited-scene")

    def test_renderer_capture_preserves_gym_contract_and_requires_all_visible_references(self):
        req, d = request(), domain()
        evidence = views(req)
        class Env:
            def observe(self):
                self._edit_validation_views = {
                    view["camera_id"]: {key: view[key] for key in ("K", "camera_to_world", "rgb", "depth", "alpha")}
                    for view in evidence}
                return {"rgb": {}}
        env = Env()
        refs = {view["camera_id"]: dict(reference_id=view["reference_id"], reference_kind=view["reference_kind"],
                                       rgb=view["reference_rgb"], depth=view["reference_depth"]) for view in evidence}
        captured = capture_local_views(env, d, req, refs, "pre")
        self.assertFalse(env.capture_edit_evidence)
        d.begin(req, ground_skin(), captured)
        del refs["right"]
        with self.assertRaisesRegex(EditTrustError, "visible camera"):
            capture_local_views(env, domain(), req, refs, "pre")

    def test_editor_cannot_mutate_outside_declared_region(self):
        from types import SimpleNamespace
        req, d = request(), domain()
        identity = d.begin(req, ground_skin(), views(req))
        env = SimpleNamespace(edit_trust_domain=d)
        require_edit_mutation(env, "insert", dict(new_bounds=req["new_bounds"]), identity)
        with self.assertRaises(EditTrustError):
            require_edit_mutation(env, "insert", dict(new_bounds=[[-2., -1., 0.], [1., 1., 2.]]), identity)


if __name__ == "__main__":
    unittest.main()
