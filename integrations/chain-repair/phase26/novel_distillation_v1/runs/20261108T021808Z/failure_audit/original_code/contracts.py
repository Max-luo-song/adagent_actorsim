"""Versioned offline view distillation. No controller or scene mutation APIs."""

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

pass  # Path is provided by the portable boundary above
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P26 = ROOT / 'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
ALLOWED = {('Background', '_features_dc'), ('Background', '_features_rest'),
           ('RoadNodes', '_features_dc'), ('RoadNodes', '_features_rest')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return _hugsim_expand(json.loads(Path(path).read_text()))


def write(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def pose(value):
    x = np.asarray(value, dtype=float)
    if x.shape != (4, 4) or not np.isfinite(x).all() or not np.allclose(x[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Invalid rigid pose')
    if not np.allclose(x[:3, :3].T @ x[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(x[:3, :3]), 1):
        raise ValueError('Invalid pose rotation')
    return x


def split_frame(index, stride=3):
    # Split by source time, shared across cameras. Hold-out frames never enter training.
    return 'eval' if index % 5 == 4 else ('train' if index % stride == 0 else 'unused')


def masked_l1(pred, target, mask):
    import torch
    if pred.shape != target.shape or pred.shape[-1] != 3 or mask.shape != pred.shape[:2]:
        raise ValueError('RGB/mask shapes disagree')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite RGB')
    if not mask.any():
        raise ValueError('Empty valid mask')
    return (pred - target).abs()[mask].mean()


def appearance_parameters(trainer):
    # Some trainer versions store models in a plain dict, outside named_parameters().
    for p in trainer.parameters():
        p.requires_grad_(False)
    for model in trainer.models.values():
        for p in model.parameters():
            p.requires_grad_(False)
    selected = {}
    for model_name, field in sorted(ALLOWED):
        model = trainer.models.get(model_name)
        if model is not None and hasattr(model, field):
            p = getattr(model, field)
            if p.numel():
                p.requires_grad_(True)
                selected[(model_name, field)] = p
    if not selected:
        raise ValueError('No supported background/road appearance parameters')
    return selected


def merge_checkpoint(base, selected):
    """Copy only approved weights into the original checkpoint, preserving all other state."""
    for (model, field), param in selected.items():
        if (model, field) not in ALLOWED:
            raise ValueError('Forbidden checkpoint update')
        old = base['models'][model][field]
        new = param.detach().cpu()
        if old.shape != new.shape or old.dtype != new.dtype or not new.isfinite().all():
            raise ValueError('Invalid replacement appearance tensor')
        base['models'][model][field] = new.clone()
    # Optimizer state is unrelated to this offline Adam instance; remove stale optimizer.
    base.pop('optimizer', None)
    return base


def check_checkpoint_invariants(before, after):
    import torch
    changed = []
    def walk(a, b, path=()):
        if path == ('optimizer',):
            return
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                if len(path) != 3 or path[0] != 'models' or tuple(path[1:]) not in ALLOWED:
                    raise ValueError('Frozen checkpoint tensor changed: ' + '.'.join(path))
                if not b.isfinite().all():
                    raise ValueError('Nonfinite appearance tensor')
                changed.append('.'.join(path))
        elif isinstance(a, dict):
            if set(a) - {'optimizer'} != set(b) - {'optimizer'}:
                raise ValueError('Checkpoint keys changed')
            for k in a:
                if k != 'optimizer':
                    walk(a[k], b[k], path + (str(k),))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError('Checkpoint sequence changed')
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + (str(i),))
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                raise ValueError('Frozen checkpoint array changed: ' + '.'.join(path))
        elif a != b:
            raise ValueError('Frozen checkpoint metadata changed: ' + '.'.join(path))
    walk(before, after)
    if not changed:
        raise ValueError('No appearance weights changed; distillation did not update model')
    return changed
