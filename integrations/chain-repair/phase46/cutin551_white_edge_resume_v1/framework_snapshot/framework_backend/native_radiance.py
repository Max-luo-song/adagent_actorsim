"""Rotate world-fixed native SH with an editing delta, without re-training.

The checkpoint evaluates C at world directions. An edit R requires C_new such
that Y(d) C_new = Y(d @ R) C_old. Never apply the full recorded root rotation:
that would change the unedited checkpoint's appearance convention.
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

pass  # Path is provided by the portable boundary above
import sys
import numpy as np


def radiance_state(model, instance):
    if not hasattr(model, '_features_rest'):
        if hasattr(model, 'get_gaussians'):
            raise ValueError('native renderer is missing SH parameters')
        return None  # Minimal geometry-only CPU contract fixtures.
    ids = model.point_ids.reshape(-1)
    if len(ids) != len(model._features_rest):
        raise ValueError('Gaussian identity and SH counts differ')
    indices = (ids == instance).nonzero(as_tuple=True)[0]
    if not len(indices):
        raise ValueError('target has no Gaussian appearance')
    rest = model._features_rest[indices].detach().clone()
    if rest.ndim != 3 or rest.shape[-1] != 3:
        raise ValueError('unsupported native SH layout')
    count = rest.shape[1] + 1
    degree = int(round(np.sqrt(count))) - 1
    if degree not in (0, 1, 2, 3) or (degree+1)**2 != count or rest.ndim != 3 or rest.shape[-1] != 3:
        raise ValueError('unsupported native SH layout')
    return indices, rest, degree


def rotated_rest(rest, degree, yaw_degrees):
    # Reuse the numerically validated degree-block rotation from asset export.
    for parent in Path(__file__).resolve().parents:
        folder = parent / 'waymo23-native-import'
        if (folder/'asset_radiance.py').is_file():
            if str(folder) not in sys.path: sys.path.insert(0, str(folder))
            break
    from asset_radiance import sh_rotation
    angle = np.deg2rad(yaw_degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
    matrix = rest.new_tensor(sh_rotation(rotation, degree)[1:,1:])
    # N x basis x RGB; no DC mixing, no change to Gaussian geometry/opacity.
    return __import__('torch').einsum('ij,njc->nic', matrix, rest)


def restore_radiance(torch, model, state):
    if state is not None:
        indices, rest, degree = state
        with torch.no_grad(): model._features_rest[indices] = rest
