"""Calibrated rigid library Gaussians merged BEFORE framework rasterization.

Asset axes and support calibration match the existing managed-asset contract.
SH radiance is evaluated in object-local directions, preserving appearance
under rigid rotation without re-fitting or accumulating coefficient rotations.
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
from dataclasses import dataclass, replace
import importlib.util
pass  # Path is provided by the portable boundary above
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path.insert(0, str(ROOT / 'waymo23-native-import'))
from asset_geometry import local_corners, validate_pose


def quat_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


_SH = None
def evaluate_sh(degree, coefficients, directions):
    global _SH
    if _SH is None:
        spec = importlib.util.spec_from_file_location('managed_asset_sh_math', ROOT / 'code/HUGSIM/utils/sh_utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SH = module.eval_sh
    return _SH(degree, coefficients.transpose(1, 2), directions)


@dataclass(frozen=True)
class ManagedAsset:
    local_means: torch.Tensor
    coefficients: torch.Tensor
    scales: torch.Tensor
    local_quats: torch.Tensor
    opacities: torch.Tensor
    pose: np.ndarray
    calibration: dict
    path: str
    degree: int

    def fields(self, camera_position):
        r = self.local_means.new_tensor(self.pose[:3, :3])
        t = self.local_means.new_tensor(self.pose[:3, 3])
        means = self.local_means @ r.T + t
        directions = (means - camera_position) @ r
        directions = torch.nn.functional.normalize(directions, dim=-1)
        colors = (evaluate_sh(self.degree, self.coefficients, directions) + .5).clamp(0, 1)
        q = Rotation.from_matrix(self.pose[:3, :3]).as_quat()[[3, 0, 1, 2]].copy()
        quats = quat_product(self.local_quats.new_tensor(q).expand_as(self.local_quats), self.local_quats)
        return dict(_means=means, _rgbs=colors, _scales=self.scales,
                    _quats=quats, _opacities=self.opacities)

    def corners(self):
        support = self.calibration['support']
        local = local_corners(support['wlh'], support['bottom_y'])
        return local @ self.pose[:3, :3].T + self.pose[:3, 3]


def load_asset(edit, device):
    params, _ = torch.load(Path(edit.asset) / 'gs.pth', map_location=device, weights_only=False)
    if len(params) != 9 or type(params[0]) is not int or params[0] != 3:
        raise ValueError('managed assets require the existing degree3 gs.pth capture schema')
    means, dc, rest, scales, quats, opacity = [params[i].detach().to(device=device, dtype=torch.float32).clone()
                                            for i in (1, 2, 3, 5, 6, 7)]
    n = len(means)
    shapes = [(n,3),(n,1,3),(n,15,3),(n,3),(n,4),(n,1)]
    for array, shape in zip((means, dc, rest, scales, quats, opacity), shapes):
        if tuple(array.shape) != shape or not torch.isfinite(array).all():
            raise ValueError('invalid managed Gaussian tensor')
    if n == 0 or torch.any(torch.linalg.vector_norm(quats, dim=-1) < 1e-8):
        raise ValueError('empty asset or invalid Gaussian rotations')
    support = edit.calibration['support']
    means[:, 0] -= support['raw_recenter_xz'][0]
    means[:, 2] -= support['raw_recenter_xz'][1]
    scales = scales.exp()
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError('invalid activated asset scales')
    return ManagedAsset(means, torch.cat((dc, rest), 1), scales,
        torch.nn.functional.normalize(quats, dim=-1), opacity.sigmoid(),
        validate_pose(edit.asset_to_world).copy(), copy.deepcopy(edit.calibration), str(edit.asset), 3)


class ManagedAssets:
    def __init__(self, trainer, device):
        self.trainer, self.device = trainer, device
        self.assets = {}
        self.label = max(trainer.gaussian_classes.values(), default=-1) + 1
        trainer.managed_dynamic_label = self.label
        self.original_collect = trainer.collect_gaussians
        trainer.collect_gaussians = self.collect

    def apply(self, edit):
        if edit.operation == 'insert':
            if edit.object_id in self.assets: raise ValueError('managed ID already exists')
            value = load_asset(edit, self.device)
        elif edit.operation == 'set_pose':
            value = replace(self.assets[edit.object_id], pose=validate_pose(edit.asset_to_world).copy())
        elif edit.operation == 'remove':
            del self.assets[edit.object_id]
            return
        else:
            raise ValueError('unsupported managed edit')
        self.assets[edit.object_id] = value

    def collect(self, *args, **kwargs):
        gs, freeze = self.original_collect(*args, **kwargs)
        self.trainer.managed_last_point_count = 0
        if not self.assets: return gs, freeze
        cam = kwargs.get('cam', args[0] if args else None)
        fields = [asset.fields(cam.camtoworlds[..., :3, 3]) for asset in self.assets.values()]
        merged = copy.copy(gs)
        for name in ('_means', '_rgbs', '_scales', '_quats', '_opacities'):
            setattr(merged, name, torch.cat([getattr(gs, name)] + [f[name] for f in fields], 0))
        count = sum(len(f['_means']) for f in fields)
        self.trainer.managed_last_point_count = count
        labels = self.trainer.pts_labels
        self.trainer.pts_labels = torch.cat((labels, labels.new_full((count,), self.label)))
        if hasattr(self.trainer, 'dynamic_pts_mask'):
            self.trainer.dynamic_pts_mask = self.trainer.pts_labels != self.trainer.gaussian_classes.get('Background', 0)
        return merged, torch.cat((freeze, freeze.new_zeros(count)))

    def clear(self): self.assets.clear()

    def close(self):
        self.clear()
        if self.trainer is not None:
            self.trainer.collect_gaussians = self.original_collect
        self.trainer = None
        self.original_collect = None
