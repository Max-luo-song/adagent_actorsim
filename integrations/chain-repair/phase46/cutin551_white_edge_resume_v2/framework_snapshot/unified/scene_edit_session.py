"""Unified edit/session lifecycle over framework or existing managed backend.

Reset restores native translations and rotations and returns a description WITHOUT rendering;
callers can apply new pre-run edits before requesting their first observation.
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

import copy
pass  # Path is provided by the portable boundary above
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'framework_backend'))
import scene_api_fwbackend as saf
from edit_api import SceneEditor


class UnifiedSceneSession(saf.SceneSession):
    def __init__(self, backend_factory=None):
        self._editor = None
        super().__init__(backend_factory=backend_factory)

    @property
    def editor(self):
        if self._editor is None:
            raise RuntimeError('load a scene first')
        self._editor.require_current()
        return self._editor

    def load_scene(self, manifest):
        super().load_scene(manifest)
        self._editor = SceneEditor(self, super().edit)
        return self.describe()

    def list_objects(self):
        return self.editor.objects()

    def edit_capabilities(self):
        return self.editor.capabilities()

    def edit_journal(self):
        return copy.deepcopy(self.editor.journal)

    def edit(self, command):
        return self.editor.edit(command)

    def describe(self):
        result = super().describe()
        if self._editor is not None:
            result.update(edit_interface='scene_edit_v1', edit_capabilities=self.edit_capabilities(),
                native_object_editing=self.editor.framework,
                full_sr_trainer=self.editor.framework,
                backend='framework_renderer' if self.editor.framework else result['backend'])
        return result

    def render(self, ego_to_world, timestamp_s):
        # Invalid pose/clock requests do not advance the editing phase.
        self._require()
        pose = saf.rigid(ego_to_world)
        saf.frame_index(timestamp_s, self.spec.fps, self.spec.frames)
        self.editor.started = True
        return super().render(pose, timestamp_s)

    def _packet(self, raw, timestamp, ego_pose):
        packet = super()._packet(raw, timestamp, ego_pose)
        editor = self.editor
        native = [dict(object_id=f'native:{group}:{index}', offset_world_m=value)
                  for (group, index), value in editor.offsets.items() if any(x != 0 for x in value)]
        native += [dict(object_id=f'native:{group}:{index}', yaw_degrees=value,
                        axis='world_z', pivot='per_frame_root_center')
                   for (group, index), value in editor.yaws.items() if value != 0]
        if editor.traffic:packet['traffic_signals']=editor.traffic.snapshot(timestamp,packet['cameras'])
        if editor.actor_paths:packet['actor_trajectories']=editor.actor_paths.snapshot()
        packet['native_edits'] = native
        packet['edit_interface'] = 'scene_edit_v1'
        packet['full_sr_trainer'] = editor.framework
        if editor.framework:
            packet['backend'] = 'framework_renderer'
        if native or (editor.traffic and editor.traffic.has_programs) or (editor.actor_paths and editor.actor_paths.programs and not editor.actor_paths.suspended):
            packet['edit_trust'].update(edited=True, status='diagnostic_unverified')
        return packet

    def reset(self):
        editor = self.editor
        editor.restore_native()
        try:
            self.backend.reset()
        except Exception:
            editor.closed = True
            raise
        if editor.actor_paths:editor.actor_paths.close()
        if editor.traffic:editor.traffic.close()
        editor.closed = True
        self.objects = {}
        self.revision = 0
        self.generation = str(uuid4())
        self._editor = SceneEditor(self, super().edit)
        return self.describe()

    def close(self):
        if self._editor is not None:
            if self._editor.actor_paths:self._editor.actor_paths.close()
            if self._editor.traffic:self._editor.traffic.close()
            self._editor.closed = True
            self._editor = None
        super().close()
