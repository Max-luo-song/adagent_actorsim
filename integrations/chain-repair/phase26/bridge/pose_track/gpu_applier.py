"""GPU-side pose applier for PoseTrackSession (imported only in the GPU harness).

Mirrors NativeSceneBackend._posed in place: recompute the SH-rotated rest
features and the native pose tensor on the ALREADY-LOADED asset model, and
update the Asset record. It deliberately does NOT go through
SceneSession.edit()/backend.apply(): no revision bump, no registry rebuild, no
UniAD history reset. Structural edits still use session.edit().

Not imported by CPU tests (requires torch/CUDA + the backend's asset record).
"""
from __future__ import annotations
import numpy as np


def make_backend_pose_applier(backend):
    """Return pose_applier(object_id, asset_to_world 4x4) for a live backend.

    backend: a NativeSceneBackend instance whose .assets maps object_id to
    Asset(model, raw_rest, source_pose, native_pose, path).
    """
    from waymo_adapter import SR_TO_HUGSIM as T
    torch = backend.torch

    def apply(object_id, pose4x4):
        asset = backend.assets[object_id]
        from asset_radiance import sh_rotation
        native = T @ np.asarray(pose4x4, dtype=float)
        matrix = sh_rotation(native[:3, :3], int(asset.model.active_sh_degree))
        with torch.no_grad():
            rest = torch.einsum('ij,njc->nic',
                                asset.raw_rest.new_tensor(matrix[1:, 1:]),
                                asset.raw_rest).contiguous()
            asset.model._features_rest = rest
            asset.native_pose = asset.model._xyz.new_tensor(native)
        asset.source_pose = np.asarray(pose4x4, dtype=float).copy()
        backend._sync_edit_metadata()

    return apply
