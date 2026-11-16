"""Bridge between PoseTrackRegistry and a SceneSession-like object.

The session adapter keeps the two event classes separate end to end:

* Edit events (insert / remove / set_trajectory) go through
  ``session.edit(...)`` (or the injected ``edit_fn``), which bumps the scene
  revision and therefore the UniAD history cache token -- correct for
  structural changes.
* Normal advancement calls the injected ``pose_applier(object_id, pose4x4)``
  for every tracked object. The applier updates ONLY the backend asset's
  render pose (and its appearance rotation) in place. It must NOT touch
  ``session.revision``/``session.generation``; the adapter asserts this.

The applier is injected so the CPU test-suite can use a NumPy mock. The GPU
harness wires it to a helper that mirrors NativeSceneBackend._posed (SH
rotation + native pose tensor update) on the already-loaded asset model.
"""
from __future__ import annotations
import numpy as np
from pose_track import PoseTrackRegistry


class PoseTrackSession:
    def __init__(self, session, pose_applier, edit_fn=None):
        self.session = session
        self.registry = PoseTrackRegistry()
        self._apply = pose_applier
        self._edit = edit_fn if edit_fn is not None else session.edit

    # -- structural edits: revision bump is intentional ----------------------
    def insert_tracked(self, object_id, trajectory, edit_request=None):
        if edit_request is not None:
            self._edit(edit_request)
        self.registry.insert(object_id, trajectory)

    def remove_tracked(self, object_id, edit_request=None):
        if edit_request is not None:
            self._edit(edit_request)
        self.registry.remove(object_id)

    def set_trajectory(self, object_id, trajectory):
        self.registry.set_trajectory(object_id, trajectory)
        pose = self.registry.render_pose(object_id)
        self._apply(object_id, pose)

    # -- normal advancement: no revision/token change allowed ----------------
    def advance_pose_track(self, timestamp_s):
        before = self._token_state()
        poses = self.registry.advance(timestamp_s)
        for oid, pose in poses.items():
            self._apply(oid, pose)
            # Keep the session-level metadata copy in sync with the pose that
            # is actually rendered/collided. This mutates ONLY the descriptive
            # record (session.objects); it must not bump revision/generation.
            objects = getattr(self.session, 'objects', None)
            if isinstance(objects, dict) and oid in objects:
                objects[oid]['asset_to_world'] = pose.tolist()
        after = self._token_state()
        if before != after:
            raise RuntimeError('normal pose advancement changed the scene '
                               'revision/generation; UniAD history would reset')
        return poses

    def _token_state(self):
        s = self.session
        return (getattr(s, 'revision', None), getattr(s, 'generation', None))

    # -- reset / scene switch: full cleanup ----------------------------------
    def reset(self):
        self.registry.reset()
        if hasattr(self.session, 'reset'):
            self.session.reset()

    # -- consistency surface for harness/audits ------------------------------
    def packet_poses(self):
        """Poses to record in the observation packet (render == collision)."""
        return {oid: np.asarray(self.registry.render_pose(oid)).copy()
                for oid in self.registry.object_ids()}
