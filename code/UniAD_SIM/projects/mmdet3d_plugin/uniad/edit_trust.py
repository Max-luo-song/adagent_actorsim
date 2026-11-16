"""N1: transactional edit admission and local revalidation, NumPy-only.

Coordinates are metric, right-handed, Z-up world coordinates. Trust is spatial
evidence, never the driving detector's class score. Reference provenance is a
caller contract, not a cryptographic claim of real-world truth.
"""

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

from dataclasses import asdict, dataclass
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above

import numpy as np

# One context across standalone and plugin imports; never enabled by metadata.
import sys
import types
from contextvars import ContextVar
from contextlib import contextmanager
_DIAGNOSTIC = sys.modules.setdefault('_hugsim_diagnostic_closed_loop_v1', types.ModuleType('_hugsim_diagnostic_closed_loop_v1'))
if not hasattr(_DIAGNOSTIC, 'enabled'):
    _DIAGNOSTIC.enabled = ContextVar('diagnostic_closed_loop', default=False)


def diagnostic_closed_loop_enabled():
    return bool(_DIAGNOSTIC.enabled.get())


@contextmanager
def diagnostic_closed_loop():
    """Explicit simulation-only suspension; never writes an N1 certificate."""
    token = _DIAGNOSTIC.enabled.set(True)
    try:
        yield
    finally:
        _DIAGNOSTIC.enabled.reset(token)


class EditTrustError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode()+array.tobytes()).hexdigest()


@dataclass(frozen=True)
class EditPolicy:
    trust_threshold: float = .8
    quarantine_confidence: float = .0
    halo_m: float = .5
    support_radius_m: float = .5
    support_error_m: float = .10
    ground_roughness_m: float = .05
    max_ground_slope_degrees: float = 20.
    min_support_points: int = 3
    min_reference_pixels: int = 16
    min_views: int = 2
    min_camera_baseline_m: float = .05
    min_cross_view_samples: int = 16
    max_cross_view_samples: int = 512
    min_valid_fraction: float = .98
    min_alpha: float = .8
    depth_absolute_error_m: float = .10
    depth_relative_error: float = .02
    max_bad_depth_fraction: float = .02
    rgb_error: float = .15
    max_bad_rgb_fraction: float = .05

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Invalid edit policy")
        if not (0 <= self.quarantine_confidence < self.trust_threshold <= 1):
            raise ValueError("Invalid trust thresholds")
        if (self.min_views < 2 or self.min_support_points < 3 or self.min_reference_pixels < 1
                or self.min_cross_view_samples < 1 or self.max_cross_view_samples < self.min_cross_view_samples
                or self.support_radius_m <= 0):
            raise ValueError("Insufficient geometric/multiview evidence requirements")
        for key in ("min_valid_fraction", "min_alpha", "max_bad_depth_fraction", "max_bad_rgb_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError("Invalid fraction: "+key)


def _points(value, minimum=1):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < minimum or not np.isfinite(p).all():
        raise EditTrustError("Expected finite metric XYZ samples")
    return p


def _bounds(value):
    b = np.asarray(value, dtype=float)
    if b.shape != (2, 3) or not np.isfinite(b).all() or np.any(b[1] <= b[0]):
        raise EditTrustError("Expected nonempty world AABB [min_xyz,max_xyz]")
    return b


def _pose(value):
    p = np.asarray(value, dtype=float)
    if (p.shape != (4, 4) or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1])
            or not np.allclose(p[:3, :3].T@p[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=1e-5)):
        raise EditTrustError("Invalid camera-to-world pose")
    return p


def _affected_roi(request, k, camera, shape, halo):
    required = np.zeros(shape, dtype=bool)
    w2c = np.linalg.inv(camera)
    for key in ("old_bounds", "new_bounds"):
        if key not in request:
            continue
        bounds = _bounds(request[key]).copy()
        bounds[0] -= halo
        bounds[1] += halo
        corners = np.array([[x, y, z, 1.] for x in bounds[:, 0]
                            for y in bounds[:, 1] for z in bounds[:, 2]])@w2c.T
        front = corners[:, 2] > 1e-3
        if not front.any():
            continue
        if not front.all():
            raise EditTrustError("Affected region crosses camera plane; supply a safer local view")
        uv = corners[:, :3]@k.T
        uv = uv[:, :2]/uv[:, 2:3]
        lo = np.maximum(np.floor(uv.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(uv.max(axis=0)).astype(int)+1, shape[::-1])
        if np.all(hi > lo):
            required[lo[1]:hi[1], lo[0]:hi[0]] = True
    return required


def check_skin(request, skin, policy):
    """Check explicit contact skin against independently observed ground.

    Not the bounding-box bottom: support points describe the actual contacts
    (e.g. wheel patches). Obstacles are independently observed occupied samples.
    """
    if request["kind"] not in ("insert", "move"):
        return dict(required=False, reason="no newly placed object")
    surface = _points(request["geometry_skin"], 4)
    contacts = _points(request["support_points"], 3)
    bounds = _bounds(request["new_bounds"])
    if (np.any(surface < bounds[0]) or np.any(surface > bounds[1])
            or np.any(contacts < bounds[0]) or np.any(contacts > bounds[1])):
        raise EditTrustError("Geometry/contact skin extends outside edit region")
    if np.linalg.matrix_rank(contacts[:, :2]-contacts[0, :2], tol=1e-5) < 2:
        raise EditTrustError("Contact samples do not span a footprint")
    ground = _points(skin["ground_points"], policy.min_support_points)
    if not skin.get("source_id") or skin.get("source_kind") not in ("sensor", "heldout"):
        raise EditTrustError("Ground skin requires observed source provenance")
    def local_height(point):
        nearby = ground[np.linalg.norm(ground[:, :2]-point[:2], axis=1) <= policy.support_radius_m]
        if len(nearby) < policy.min_support_points:
            raise EditTrustError("Insufficient ground coverage at a contact patch")
        xy = nearby[:, :2]-point[:2]
        design = np.column_stack([xy, np.ones(len(xy))])
        if np.linalg.matrix_rank(design) < 3:
            raise EditTrustError("Ground samples cannot constrain a local plane")
        a, b, height = np.linalg.lstsq(design, nearby[:, 2], rcond=None)[0]
        if np.sqrt(np.mean((design@[a, b, height]-nearby[:, 2])**2)) > policy.ground_roughness_m:
            raise EditTrustError("Ground skin is rough or inconsistent")
        if np.degrees(np.arctan(np.hypot(a, b))) > policy.max_ground_slope_degrees:
            raise EditTrustError("Ground support slope exceeds shared policy")
        return float(height)

    residuals = []
    for contact in contacts:
        residuals.append(float(contact[2]-local_height(contact)))
        if abs(residuals[-1]) > policy.support_error_m:
            raise EditTrustError("Floating or penetrating contact skin")
    occupied = _points(skin.get("occupied_points", np.empty((0, 3))), 0)
    for point in surface:
        if point[2] < local_height(point)-policy.support_error_m:
            raise EditTrustError("Object skin penetrates observed ground")
    # Conservative AABB collision rejection. An OBB/SDF backend may replace it
    # without changing the admission lifecycle; ground samples are separate.
    interior = ((occupied > bounds[0]+policy.support_error_m)
                & (occupied < bounds[1]-policy.support_error_m)).all(axis=1)
    if interior.any():
        raise EditTrustError("Placed object intersects observed occupied skin")
    return dict(required=True, contacts=len(contacts), contact_residuals_m=residuals,
                ground_source=skin["source_id"], geometry_hash=array_digest(surface),
                ground_hash=array_digest(ground), occupied_hash=array_digest(occupied))


def check_views(views, request, stage, revision, policy, scene_digest=None):
    """Measure reference agreement and bidirectional depth reprojection.

    A renderer compared with itself is not independent evidence. Every view
    supplies calibrated camera-Z depth, RGB [0,1], alpha, affected ROI and an
    independently sourced reference. For erase/repair only sensor/held-out
    references are admitted: hole filling cannot certify its own hallucination.
    """
    if len(views) < policy.min_views:
        raise EditTrustError("At least two real calibrated views are required")
    parsed, metrics, ids = [], [], set()
    for view in views:
        if (view.get("request_hash") != digest(request) or view.get("stage") != stage
                or view.get("revision") != revision
                or view.get("scene_digest") != (scene_digest or request["scene_digest"])):
            raise EditTrustError("Stale or mismatched edit evidence")
        identity = view.get("camera_id")
        if not identity or identity in ids:
            raise EditTrustError("Duplicate or missing camera identity")
        ids.add(identity)
        allowed = ("sensor", "heldout") if request["kind"] in ("erase", "repair") else ("sensor", "heldout", "independent_geometry")
        if (view.get("reference_kind") not in allowed or not view.get("reference_id")
                or not view.get("render_id") or view["reference_id"] == view["render_id"]):
            raise EditTrustError("Missing independent reference provenance")
        depth = np.asarray(view["depth"], dtype=float)
        reference = np.asarray(view["reference_depth"], dtype=float)
        roi = np.asarray(view["roi"], dtype=bool)
        alpha = np.asarray(view["alpha"], dtype=float)
        rgb, ref_rgb = (np.asarray(view[key], dtype=float) for key in ("rgb", "reference_rgb"))
        if (depth.ndim != 2 or reference.shape != depth.shape or roi.shape != depth.shape
                or alpha.shape != depth.shape or rgb.shape != (*depth.shape, 3) or ref_rgb.shape != rgb.shape):
            raise EditTrustError("Mismatched local-render evidence arrays")
        k = np.asarray(view["K"], dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise EditTrustError("Invalid pinhole calibration")
        camera = _pose(view["camera_to_world"])
        required_roi = _affected_roi(request, k, camera, depth.shape, policy.halo_m)
        if int(required_roi.sum()) < policy.min_reference_pixels or np.any(required_roi & ~roi):
            raise EditTrustError("Evidence ROI omits affected edit region or its halo")
        # Score only the projected edit+halo. A caller cannot dilute local
        # hallucinations by supplying the whole image as an oversized ROI.
        roi = required_roi
        if not (np.isfinite(reference[roi]).all() and (reference[roi] > 0).all()
                and np.isfinite(ref_rgb[roi]).all() and ((ref_rgb[roi] >= 0) & (ref_rgb[roi] <= 1)).all()):
            raise EditTrustError("Affected ROI lacks observed reference coverage")
        count = int(roi.sum())
        if count < policy.min_reference_pixels:
            raise EditTrustError("Insufficient affected ROI reference pixels")
        valid = (np.isfinite(depth) & (depth > 0) & np.isfinite(alpha)
                 & (alpha >= policy.min_alpha) & (alpha <= 1))
        fraction = float(valid[roi].mean())
        if fraction < policy.min_valid_fraction:
            raise EditTrustError("Hole/low-opacity/nonfinite pixels in edited ROI")
        tolerance = policy.depth_absolute_error_m+policy.depth_relative_error*reference
        bad_depth = float((~valid[roi] | (np.abs(depth[roi]-reference[roi]) > tolerance[roi])).mean())
        rgb_ok = np.isfinite(rgb).all(axis=-1) & ((rgb >= 0) & (rgb <= 1)).all(axis=-1)
        bad_rgb = float((~rgb_ok[roi] | (np.max(np.abs(rgb[roi]-ref_rgb[roi]), axis=-1) > policy.rgb_error)).mean())
        if bad_depth > policy.max_bad_depth_fraction or bad_rgb > policy.max_bad_rgb_fraction:
            raise EditTrustError("Depth/RGB reference disagreement: ghosting or inconsistent repair")
        if request["kind"] in ("insert", "move"):
            surface = _points(request["geometry_skin"], 4)
            camera_points = np.column_stack([surface, np.ones(len(surface))])@np.linalg.inv(camera).T
            projected = camera_points[:, :3]@k.T
            uv = np.rint(projected[:, :2]/np.maximum(camera_points[:, 2:3], 1e-6)).astype(int)
            h, w = depth.shape
            inside = ((camera_points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            idx = np.flatnonzero(inside)
            measured = depth[uv[idx, 1], uv[idx, 0]]
            tolerance = policy.depth_absolute_error_m+policy.depth_relative_error*camera_points[idx, 2]
            if not np.any(np.abs(measured-camera_points[idx, 2]) <= tolerance):
                raise EditTrustError("Rendered edit is not supported by the declared geometry skin")
        parsed.append(dict(depth=depth, roi=roi, valid=valid, K=k, pose=camera))
        metrics.append(dict(camera_id=identity, valid_fraction=fraction, bad_depth_fraction=bad_depth,
                            bad_rgb_fraction=bad_rgb, reference_kind=view["reference_kind"],
                            reference_id=view["reference_id"], depth_hash=array_digest(depth),
                            reference_depth_hash=array_digest(reference), rgb_hash=array_digest(rgb),
                            reference_rgb_hash=array_digest(ref_rgb), roi_hash=array_digest(roi),
                            scene_digest=view["scene_digest"], revision=revision,
                            calibration_hash=digest(dict(K=k.tolist(), pose=camera.tolist()))))
    for i, a in enumerate(parsed):
        if not any(np.linalg.norm(a["pose"][:3, 3]-b["pose"][:3, 3]) >= policy.min_camera_baseline_m
                   for j, b in enumerate(parsed) if i != j):
            raise EditTrustError("Duplicated viewpoints do not establish multiview geometry")
    pair_metrics = []
    for i, source in enumerate(parsed):
        yy, xx = np.nonzero(source["roi"] & source["valid"])
        ids = np.linspace(0, len(xx)-1, min(len(xx), policy.max_cross_view_samples)).astype(int)
        pixels = np.stack([xx[ids], yy[ids], np.ones(len(ids))], axis=-1)
        xyz = (pixels@np.linalg.inv(source["K"]).T)*source["depth"][yy[ids], xx[ids], None]
        world = np.column_stack([xyz, np.ones(len(xyz))])@source["pose"].T
        matches, disagreements = 0, 0
        for j, target in enumerate(parsed):
            if i == j:
                continue
            camera = world@np.linalg.inv(target["pose"]).T
            projected = camera[:, :3]@target["K"].T
            z = camera[:, 2]
            uv = np.rint(projected[:, :2]/np.maximum(z[:, None], 1e-6)).astype(int)
            h, w = target["depth"].shape
            inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            idx = np.flatnonzero(inside)
            x, y = uv[idx, 0], uv[idx, 1]
            idx = idx[target["roi"][y, x]]
            x, y = uv[idx, 0], uv[idx, 1]
            measured = target["depth"][y, x]
            tol = policy.depth_absolute_error_m+policy.depth_relative_error*z[idx]
            # Real foreground occlusion is not a disagreement with a hidden point.
            occluded = target["valid"][y, x] & (measured < z[idx]-tol)
            eligible = ~occluded
            matches += int(eligible.sum())
            disagreements += int((eligible & (~target["valid"][y, x] | (np.abs(measured-z[idx]) > tol))).sum())
        if matches < policy.min_cross_view_samples or disagreements/max(matches, 1) > policy.max_bad_depth_fraction:
            raise EditTrustError("Insufficient overlap or inconsistent multiview surface")
        pair_metrics.append(dict(source_view=i, comparable_samples=matches, disagreements=disagreements))
    return dict(views=metrics, reprojection=pair_metrics,
                provenance_verified_externally=False, reference_contract="independently supplied local evidence")


class EditTrustDomain:
    """Persistent spatial quarantine with conservative scene-wide eval gate."""
    def __init__(self, confidence, observations, origin_xy, cell_size_m, source_id,
                 scene_digest, policy=None):
        self.policy = EditPolicy(**(policy or {}))
        self.confidence = np.asarray(confidence, dtype=float).copy()
        self.observations = np.asarray(observations, dtype=int).copy()
        self.origin = np.asarray(origin_xy, dtype=float)
        self.cell_size = float(cell_size_m)
        if (self.confidence.ndim != 2 or not self.confidence.size or self.observations.shape != self.confidence.shape
                or not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1))
                or self.origin.shape != (2,) or not np.isfinite(self.origin).all()
                or not np.isfinite(self.cell_size) or self.cell_size <= 0 or not source_id or not scene_digest):
            raise EditTrustError("Invalid baseline trust field/provenance")
        self.source_id, self.scene_digest = str(source_id), str(scene_digest)
        self.revision = 0
        self.edits = {}

    def cells(self, request):
        boxes = [_bounds(request[key]) for key in ("old_bounds", "new_bounds") if key in request]
        if not boxes:
            raise EditTrustError("Missing affected region")
        low = np.min([b[0, :2] for b in boxes], axis=0)-self.policy.halo_m
        high = np.max([b[1, :2] for b in boxes], axis=0)+self.policy.halo_m
        shape_xy = np.array(self.confidence.shape[::-1])
        if np.any(low < self.origin) or np.any(high >= self.origin+shape_xy*self.cell_size):
            raise EditTrustError("Affected region/halo extends outside known trust domain")
        lo = np.floor((low-self.origin)/self.cell_size).astype(int)
        hi = np.floor((high-self.origin)/self.cell_size).astype(int)
        return [(int(y), int(x)) for y in range(lo[1], hi[1]+1) for x in range(lo[0], hi[0]+1)]

    def begin(self, request, skin, pre_views):
        if any(edit["state"] != "verified" for edit in self.edits.values()):
            raise EditTrustError("Finish or revalidate the outstanding edit before another mutation")
        request = _hugsim_expand(json.loads(json.dumps(request, allow_nan=False)))
        kind = request.get("kind")
        if kind not in ("insert", "move", "erase", "repair") or not request.get("edit_id"):
            raise EditTrustError("Invalid edit operation/identity")
        if request["edit_id"] in self.edits:
            raise EditTrustError("Edit identity has already been used")
        required = ("new_bounds",) if kind == "insert" else (("old_bounds", "new_bounds") if kind == "move" else ("old_bounds",))
        if any(key not in request for key in required) or request.get("scene_digest") != self.scene_digest:
            raise EditTrustError("Missing edit bounds or mismatched scene digest")
        cells = self.cells(request)
        if any(self.confidence[y, x] < self.policy.trust_threshold or self.observations[y, x] < self.policy.min_views for y, x in cells):
            raise EditTrustError("Edit affects unknown, quarantined, or insufficiently observed cells")
        geometry = check_skin(request, skin, self.policy)
        visual = check_views(pre_views, request, "pre", self.revision, self.policy)
        previous = [float(self.confidence[y, x]) for y, x in cells]
        trust_evidence = dict(min_confidence=min(previous),
                              min_observations=min(int(self.observations[y, x]) for y, x in cells),
                              source_id=self.source_id, field_hash=array_digest(self.confidence))
        self.revision += 1
        for y, x in cells:
            self.confidence[y, x] = self.policy.quarantine_confidence
        self.edits[request["edit_id"]] = dict(request=request, request_hash=digest(request), cells=cells,
            previous_confidence=previous, state="quarantined", revision=self.revision,
            pre_trust=trust_evidence,
            pre_geometry=geometry, pre_visual=visual, post_visual=None, mutation_complete=False)
        return request["edit_id"]

    def apply(self, edit_id, mutation, resulting_scene_digest, persist_path):
        edit = self.edits[edit_id]
        if edit["state"] != "quarantined" or edit["mutation_complete"]:
            raise EditTrustError("Edit is not an unapplied quarantined transaction")
        if not resulting_scene_digest or resulting_scene_digest == self.scene_digest:
            raise EditTrustError("Edited scene must have a new content/version digest")
        # Persist quarantine before touching the scene. A crash or partial
        # mutation cannot leave a trusted certificate on disk.
        self.save(persist_path)
        try:
            actual_digest = mutation()
            if actual_digest != resulting_scene_digest:
                raise EditTrustError("Mutation readback digest differs from the declared edited scene")
        except BaseException:
            edit["state"] = "mutation_failed"
            self.save(persist_path)
            raise
        self.scene_digest = str(resulting_scene_digest)
        edit["mutation_complete"] = True
        edit["resulting_scene_digest"] = self.scene_digest
        self.save(persist_path)

    def revalidate(self, edit_id, post_views, persist_path):
        edit = self.edits[edit_id]
        if not edit["mutation_complete"] or edit["state"] not in ("quarantined", "revalidation_failed"):
            raise EditTrustError("No applied edit awaiting local revalidation")
        if edit["resulting_scene_digest"] != self.scene_digest:
            raise EditTrustError("Another scene mutation invalidated this edit's evidence")
        try:
            visual = check_views(post_views, edit["request"], "post", self.revision, self.policy,
                                 self.scene_digest)
        except (EditTrustError, ValueError, KeyError) as exc:
            edit["state"] = "revalidation_failed"
            edit["failure"] = str(exc)
            self.save(persist_path)
            raise
        edit["post_visual"], edit["state"] = visual, "verified"
        for (y, x), previous in zip(edit["cells"], edit["previous_confidence"]):
            self.confidence[y, x] = previous  # Never inflate the baseline trust.
        self.save(persist_path)

    def assert_evaluable(self):
        if any(edit["state"] != "verified" for edit in self.edits.values()):
            raise EditTrustError("Edited scene is quarantined; local revalidation required")

    def state(self):
        value = dict(schema_version=1, coordinate_system="world_z_up_meters",
                     policy=asdict(self.policy), confidence=self.confidence.tolist(),
                     observations=self.observations.tolist(), origin_xy=self.origin.tolist(),
                     cell_size_m=self.cell_size, source_id=self.source_id,
                     scene_digest=self.scene_digest, revision=self.revision, edits=self.edits)
        return dict(value, state_hash=digest(value))

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix+".tmp")
        with tmp.open("w") as stream:
            json.dump(self.state(), stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @classmethod
    def load(cls, path):
        value = _hugsim_expand(json.loads(Path(path).read_text()))
        expected = value.pop("state_hash")
        if digest(value) != expected or value.get("schema_version") != 1:
            raise EditTrustError("Corrupt/stale trust state")
        domain = cls(value["confidence"], value["observations"], value["origin_xy"],
                     value["cell_size_m"], value["source_id"], value["scene_digest"], value["policy"])
        domain.revision, domain.edits = value["revision"], value["edits"]
        return domain

    def evaluation_metadata(self):
        self.assert_evaluable()
        return self.runtime_metadata()

    def runtime_metadata(self):
        state = self.state()
        # Packets carry an audit certificate, not the full spatial trust raster.
        # The complete field remains in the atomic on-disk state.
        domain_hash = state.pop("state_hash")
        for key in ("confidence", "observations"):
            state.pop(key)
        state["domain_state_hash"] = domain_hash
        state["state_hash"] = digest(state)
        return dict(edited=bool(self.edits), scene_digest=self.scene_digest,
                    trust_state=state)


def assert_edit_evaluable(metadata, expected_scene_digest=None):
    """One admission contract for simulator, model worker and replay runner."""
    if diagnostic_closed_loop_enabled() and expected_scene_digest is None:
        return
    declared = any(bool(metadata.get(key)) for key in (
        "scene_edits", "edited_scene", "remove_rigid", "asset_motion", "inserted_asset_present"))
    info = metadata.get("edit_trust")
    if info is None:
        if declared or expected_scene_digest is not None:
            raise EditTrustError("Edited scene has no N1 trust evidence")
        return
    if not info.get("edited"):
        if declared or expected_scene_digest is not None or (info.get("trust_state") or {}).get("edits"):
            raise EditTrustError("Conflicting edited-scene metadata")
        return
    state = dict(info.get("trust_state") or {})
    expected = state.pop("state_hash", None)
    if (not expected or digest(state) != expected or state.get("schema_version") != 1
            or info.get("scene_digest") != state.get("scene_digest") or not state.get("edits")):
        raise EditTrustError("Missing, mismatched, or corrupt N1 edit certificate")
    if expected_scene_digest is not None and expected_scene_digest != state["scene_digest"]:
        raise EditTrustError("N1 certificate belongs to a different scene/edit binding")
    policy = EditPolicy(**state["policy"])
    for edit in state["edits"].values():
        if (edit.get("state") != "verified" or not edit.get("mutation_complete")
                or not edit.get("post_visual") or not edit.get("pre_visual")
                or digest(edit["request"]) != edit.get("request_hash")):
            raise EditTrustError("Unverified edit cannot enter closed-loop evaluation")
        trust = edit.get("pre_trust") or {}
        if (trust.get("min_confidence", 0) < policy.trust_threshold
                or trust.get("min_observations", 0) < policy.min_views or not trust.get("source_id")):
            raise EditTrustError("Certificate has no trusted baseline-region evidence")
        for visual, expected_digest in ((edit["pre_visual"], edit["request"]["scene_digest"]),
                                        (edit["post_visual"], edit["resulting_scene_digest"])):
            views = visual.get("views", [])
            pairs = visual.get("reprojection", [])
            if len(views) < policy.min_views or len(pairs) < policy.min_views:
                raise EditTrustError("Certificate lacks measured multiview evidence")
            for view in views:
                if (view.get("valid_fraction", 0) < policy.min_valid_fraction
                        or view.get("bad_depth_fraction", 1) > policy.max_bad_depth_fraction
                        or view.get("bad_rgb_fraction", 1) > policy.max_bad_rgb_fraction
                        or not view.get("reference_id")
                        or view.get("scene_digest") != expected_digest
                        or any(len(view.get(key, "")) != 64 for key in (
                            "depth_hash", "reference_depth_hash", "rgb_hash", "reference_rgb_hash", "roi_hash", "calibration_hash"))):
                    raise EditTrustError("Certificate contains invalid local measurements")
            for pair in pairs:
                if (pair.get("comparable_samples", 0) < policy.min_cross_view_samples
                        or pair.get("disagreements", float("inf"))/max(pair.get("comparable_samples", 0), 1) > policy.max_bad_depth_fraction):
                    raise EditTrustError("Certificate contains inconsistent multiview measurements")


def mark_unverified_edit(env, operation):
    """Legacy preview adapter: quarantine marker, never a valid certificate."""
    if not hasattr(env, "_unverified_scene_edits"):
        env._unverified_scene_edits = []
    env._unverified_scene_edits.append(operation)
