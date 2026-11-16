"""One actual scene session: reset a loaded scene or load another scene safely."""

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

import hashlib
import json
pass  # Path is provided by the portable boundary above

from scene_edit_session import UnifiedSceneSession,saf


class SharedWeatherScene:
    def __init__(self,folder,backend_factory=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.backend_factory=backend_factory
        self.loads=0;self.key=None;self.active=False;self.failed=False;self.transitions=[]
        self.session=UnifiedSceneSession(self._make_backend)

    def _make_backend(self,spec):
        out=self.folder/f'backend_load_{self.loads}';self.loads+=1
        if self.backend_factory is not None:return self.backend_factory(spec,out)
        import torch
        # The old scene has been closed before this factory is called.
        torch.cuda.empty_cache()
        from runtime_backend import RuntimeWeatherBackend
        return RuntimeWeatherBackend(spec,out)

    def acquire(self,manifest,weather_output):
        if self.active or self.failed:raise RuntimeError('Shared scene is active or failed')
        spec=saf.SceneSpec.read(manifest)
        key=hashlib.sha256(json.dumps(dict(source=spec.source,fps=spec.fps,scale=spec.scale,
            camera_order=list(spec.camera_order)),sort_keys=True).encode()).hexdigest()
        old_backend=self.session.backend;old_editor=self.session._editor
        same=old_backend is not None and key==self.key
        try:
            if same:self.session.reset()
            else:self.session.load_scene(manifest)
            if old_editor is not None:
                try:old_editor.objects()
                except RuntimeError:pass
                else:raise RuntimeError('Old scene editor remained valid')
            self.session.backend.begin_weather_episode(weather_output)
            transition=dict(scene=spec.source['scene'],same_backend_reset=same,
                scene_switched=old_backend is not None and not same,
                old_backend_closed=old_backend is None or same or old_backend.closed,
                session_object_id=id(self.session),backend_object_id=id(self.session.backend),
                generation=self.session.generation,weather_epoch=self.session.backend.weather_epoch,
                checkpoint=spec.source['checkpoint'])
            transition['gaussian_points_by_group']={name:int(model._means.shape[0])
                for name,model in self.session.backend.trainer.models.items() if hasattr(model,'_means')}
            self.transitions.append(transition)
            (self.folder/'transitions.json').write_text(json.dumps(self.transitions,indent=2))
            self.key=key;self.active=True
            return self.session,transition
        except BaseException:
            self.failed=True
            raise

    def release(self):
        if not self.active:return
        self.active=False
        if (self.session.revision!=0 or self.session.edit_journal()
                or self.session.backend._last is not None):
            self.failed=True
            raise RuntimeError('Driving episode did not reset its shared scene before release')

    def close(self):self.session.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
