"""Diagnostic pose-aligned dynamic coverage, using full-scene occlusion.

No source renderer changes: rerasterize the SAME Gaussian collection/camera
with dynamic RGB=1, static RGB=0, preserving opacity/depth/occlusion.
Not ground-truth segmentation; excludes RoadNodes. TrafficLightNodes are
protected only when explicitly requested by the unified weather adapter.
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
import sys
pass  # Path is provided by the portable boundary above
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'framework_backend'))
from backend import FrameworkSceneBackend
GROUPS=('RigidNodes','SMPLNodes','DeformableNodes')


def install_mask_capture(trainer,protect_signals=False):
    original=trainer.render_gaussians
    captured=[]
    def render(gs,cam,**kwargs):
        outputs,fn=original(gs=gs,cam=cam,**kwargs)
        info=trainer.info
        semantic=copy.copy(gs)
        labels=trainer.pts_labels
        dynamic=labels.new_zeros(labels.shape,dtype=torch.bool)
        for name in GROUPS+(('TrafficLightNodes',) if protect_signals else ()):
            if name in trainer.gaussian_classes:dynamic |= labels==trainer.gaussian_classes[name]
        managed_label = getattr(trainer, 'managed_dynamic_label', None)
        if managed_label is not None: dynamic |= labels == managed_label
        semantic._rgbs=dynamic[:,None].expand(-1,3).to(gs.rgbs.dtype)
        try:
            mask_outputs,_=original(gs=semantic,cam=cam,**kwargs)
            value=mask_outputs['rgb_gaussians'].detach().cpu().numpy()
            captured.append(np.clip(value.mean(-1),0,1).astype(np.float32))
        finally:trainer.info=info
        return outputs,fn
    trainer.render_gaussians=render
    return captured


class MaskFrameworkBackend(FrameworkSceneBackend):
    def __init__(self,spec,protect_signals=False):
        super().__init__(spec)
        self._plain_raster=self.trainer.render_gaussians
        self._coverage=install_mask_capture(self.trainer,protect_signals=protect_signals)
    def render(self,pose,timestamp):
        self._coverage.clear()
        raw=super().render(pose,timestamp)
        names=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']
        if len(self._coverage)!=3:raise RuntimeError('expected one semantic raster per camera')
        for name,coverage in zip(names,self._coverage):
            cam=raw['cameras'][name]
            if coverage.shape!=cam['rgb'].shape[:2] or not np.isfinite(coverage).all():
                raise ValueError('invalid dynamic coverage')
            cam['weather_dynamic_coverage']=coverage
            cam['weather_dynamic_mask']=(coverage>.2).astype(np.uint8)*255
            cam['weather_mask_source']='same-scene semantic coverage; full static occlusion; threshold .2'
        return raw

    def close(self):
        # Bound raster methods retain the full trainer/checkpoint. Release both
        # before a replacement scene is loaded in this same Python process.
        trainer = getattr(self, 'trainer', None)
        plain = getattr(self, '_plain_raster', None)
        if trainer is not None and plain is not None:
            trainer.render_gaussians = plain
        self._plain_raster = None
        coverage = getattr(self, '_coverage', None)
        if coverage is not None:
            coverage.clear()
        del trainer, plain
        super().close()
