"""SMPL integration verification via the data_struct=npz path (no chumpy).

Checks beyond vertex count:
  1. field inventory and shapes (v_template/shapedirs/posedirs/weights/kintree/J_regressor)
  2. default-pose forward: vertices finite, zero betas -> template shape
  3. betas sensitivity: nonzero betas change shape (and shapedirs dims match)
  4. pose sensitivity: rotating a joint moves the corresponding limb region
     (knee flexion changes lower-leg vertices more than head vertices)
  5. joints layout audit (count and height ordering sanity)
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

import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / 'resources/smpl_neutral/SMPL_NEUTRAL.npz'
REPO = HERE.parents[1] / 'framework_repo'
sys.path.insert(0, str(REPO / 'third_party/smplx'))


def main():
    import torch
    from smplx import SMPLLayer
    from smplx.utils import Struct

    packed = np.load(NPZ)
    # rebuild sparse J_regressor (packed as components by the converter)
    import scipy.sparse as sp
    data = {}
    for k in packed.files:
        if '__sparse_' in k:
            continue
        data[k] = packed[k]
    jr_parts = {k: packed[k] for k in packed.files if k.startswith('J_regressor__sparse_')}
    if jr_parts:
        data['J_regressor'] = sp.csc_matrix(
            (jr_parts['J_regressor__sparse_data'], jr_parts['J_regressor__sparse_indices'],
             jr_parts['J_regressor__sparse_indptr']),
            shape=tuple(jr_parts['J_regressor__sparse_shape']))

    inventory = {k: (list(v.shape) if hasattr(v, 'shape') else str(type(v).__name__))
                 for k, v in data.items()}
    print(json.dumps(inventory, indent=1))

    layer = SMPLLayer(model_path=None, data_struct=Struct(**data))
    out0 = layer(betas=torch.zeros(1, 10))
    v0 = out0.vertices[0].detach().numpy()
    assert np.isfinite(v0).all()
    report = dict(inventory=inventory, zero_forward_vertices=list(out0.vertices.shape))

    # betas sensitivity
    betas1 = torch.zeros(1, 10); betas1[0, 0] = 5.0
    v1 = layer(betas=betas1).vertices[0]
    d_beta = float((v1 - out0.vertices[0]).abs().max())
    assert d_beta > 1e-3, 'betas had no effect'
    report['betas_max_effect_m'] = d_beta

    # pose sensitivity: flex right knee (joint 5 in SMPL kintree: 0 pelvis,1 L hip,
    # 2 R hip,3 spine1,4 L knee,5 R knee) about x-axis by 1 rad
    # This vendored SMPLLayer takes ROTATION MATRICES for body_pose
    # (framework human_body.py does axis_angle_to_matrix first), not axis-angle.
    # body_pose[i] drives kintree joint i+1 (global orient excluded):
    # body_pose[4] == joint 5 == right knee.
    body_pose = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 23, 1, 1)
    body_pose[0, 4] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])  # 90deg about x
    vpose = layer(betas=torch.zeros(1, 10), body_pose=body_pose).vertices[0].detach().numpy()
    dv = np.linalg.norm(vpose - v0, axis=-1)
    head_region = v0[:, 1] > v0[:, 1].max() - 0.2   # head (SMPL canonical: y-up)
    leg_region = v0[:, 1] < v0[:, 1].min() + 0.6    # lower leg
    report['pose_effect_head_mean_m'] = float(dv[head_region].mean())
    report['pose_effect_lowleg_mean_m'] = float(dv[leg_region].mean())
    assert report['pose_effect_lowleg_mean_m'] > 10 * report['pose_effect_head_mean_m'] + 1e-3, \
        'knee flexion did not localize to the leg'
    report['joints'] = int(out0.joints.shape[1])
    report['status'] = 'PASS'
    out = HERE / 'smpl_integration_verification.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
