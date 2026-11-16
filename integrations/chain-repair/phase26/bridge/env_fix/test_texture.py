"""nvdiffrast functional smoke: the EXACT call used by framework models/modules.py.

models/modules.py (CubeMapLighting.forward) calls:
    dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
This script builds the same shapes/dtypes and checks output finiteness and a
known-content lookup (constant cubemap must return that constant everywhere).

Version deviation policy: framework pins diffrp-nvdiffrast==0.3.3.1 (fork,
source unavailable offline); this test runs against the OFFICIAL nvdiffrast
source available locally. A pass means API-compatible for this call only;
the deviation is recorded, not hidden.
"""
import sys
import torch
import nvdiffrast.torch as dr

print('nvdiffrast version:', getattr(dr, '__version__', None) or __import__('nvdiffrast').__version__)

res = 8
base = torch.full((6, res, res, 3), 0.5, device='cuda')  # same as modules.py init
l = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]],
                 device='cuda')  # viewdirs like image_infos['viewdirs']
to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                         dtype=torch.float32, device='cuda')
lg = (l @ to_opengl.T).contiguous()
lg = lg.reshape(1, 1, -1, 3)  # cube mode requires [N,H,W,3]; mirrors modules.py
light = dr.texture(base[None, ...], lg, filter_mode='linear', boundary_mode='cube')
assert torch.isfinite(light).all(), 'non-finite texture output'
assert torch.allclose(light, torch.full_like(light, 0.5), atol=1e-5), \
    'constant cubemap lookup mismatch: %s' % light.flatten()[:6]
print('constant-cubemap lookup OK; out shape', tuple(light.shape))

# non-constant: face +x painted red
base2 = base.clone(); base2[0, :, :] = torch.tensor([1., 0., 0.], device='cuda')
light2 = dr.texture(base2[None, ...], lg, filter_mode='linear', boundary_mode='cube')
px = light2.flatten()[0:3]  # first viewdir (+x) result
print('non-constant probe (+x viewdir):', px.flatten()[:3].tolist())
print('SMOKE OK')
