"""Bridge legacy Gaussian editors and the shared N1 trust contract."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"))
from edit_trust import EditTrustDomain, EditTrustError, assert_edit_evaluable, mark_unverified_edit, digest, diagnostic_closed_loop


def asset_scene_binding(checkpoint, asset_report):
    """Cheap immutable-input fingerprint; no large checkpoint traversal.

    This detects accidental file/placement changes, not adversarial replacement
    preserving file metadata. A content-addressed asset store may supply stronger
    source identifiers at the same interface.
    """
    def fingerprint(path):
        path = Path(path).resolve()
        stat = path.stat()
        return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    placement = next(p for p in asset_report["placements"] if p["case"] == "heading_A")
    return digest(dict(checkpoint=fingerprint(checkpoint),
                       asset=fingerprint(Path(asset_report["asset"])/"gs.pth"),
                       placement=placement, recenter=asset_report["local_recenter_xz"]))


def load_evaluation_domain(path, expected_binding):
    domain = EditTrustDomain.load(path)
    assert_edit_evaluable(dict(edit_trust=domain.evaluation_metadata()), expected_binding)
    return domain


def swept_instance_bounds(state, instance, translation=None):
    positions = state["instances_trans"][:, instance].detach().cpu().numpy().copy()
    if translation is not None:
        positions += np.asarray(translation)
    # Conservative for arbitrary instance rotation over the edited sequence.
    radius = float(np.linalg.norm(state["instances_size"][instance].detach().cpu().numpy())/2)
    return np.stack([positions.min(axis=0)-radius, positions.max(axis=0)+radius])


def require_edit_mutation(env, kind, bounds, edit_id=None, preview_only=False):
    if preview_only:
        mark_unverified_edit(env, dict(kind=kind, bounds={k: np.asarray(v).tolist() for k, v in bounds.items()},
                                       status="diagnostic_preview_only"))
        return
    domain = getattr(env, "edit_trust_domain", None)
    if domain is None or edit_id not in domain.edits:
        raise EditTrustError("N1 preflight required: attach edit_trust_domain and provide edit_id; diagnostic previews require preview_only=True")
    edit = domain.edits[edit_id]
    if edit["state"] != "quarantined" or edit["mutation_complete"] or edit["request"]["kind"] != kind:
        raise EditTrustError("Edit has no active preflight/quarantine transaction")
    for key, actual in bounds.items():
        declared = np.asarray(edit["request"].get(key), dtype=float)
        actual = np.asarray(actual, dtype=float)
        if declared.shape != (2, 3) or np.any(actual[0] < declared[0]) or np.any(actual[1] > declared[1]):
            raise EditTrustError("Actual swept edit exceeds the preflight region")


def runtime_edit_metadata(env):
    unverified = getattr(env, "_unverified_scene_edits", [])
    if unverified:
        return dict(edited=True, status="quarantined", operations=unverified)
    domain = getattr(env, "edit_trust_domain", None)
    if domain is not None:
        return domain.runtime_metadata()
    return dict(edited=False)


def capture_local_views(env, domain, request, independent_references, stage):
    """Collect real renderer RGB/ED/alpha in first-ego-relative Z-up coordinates.

    Independent references must be provided for every captured camera that sees
    the edit+halo. No detector output or generated edit image is a reference.
    The Gym observation structure is unchanged; evidence capture is opt-in.
    """
    from uuid import uuid4
    from edit_trust import _affected_roi
    previous = getattr(env, "capture_edit_evidence", False)
    env.capture_edit_evidence = True
    env._edit_validation_views = {}
    try:
        env.observe()
    finally:
        env.capture_edit_evidence = previous
    if not env._edit_validation_views:
        raise EditTrustError("Renderer did not expose local RGB/depth/alpha evidence")
    capture_id = str(uuid4())
    views = []
    for name, rendered in env._edit_validation_views.items():
        required = _affected_roi(request, rendered["K"], rendered["camera_to_world"],
                                 rendered["depth"].shape, domain.policy.halo_m)
        if int(required.sum()) < domain.policy.min_reference_pixels:
            continue
        if name not in independent_references:
            raise EditTrustError("Missing independent local reference for visible camera: "+name)
        reference = independent_references[name]
        views.append(dict(rendered, camera_id=name, render_id=capture_id+":"+name,
                          reference_id=reference["reference_id"], reference_kind=reference["reference_kind"],
                          reference_rgb=reference["rgb"], reference_depth=reference["depth"],
                          roi=required, stage=stage, request_hash=digest(request),
                          revision=domain.revision, scene_digest=domain.scene_digest))
    return views
