"""CPU validation of an exported managed asset against the native render chain.

Checks (all diagnostic, no certification):
1. Round-trip geometry: managed placement (calibration recenter + trajectory pose)
   reproduces scene_math.rigid_geometry world points for every visible frame.
2. Colour chain: at the SH reference frame the managed chain (stored
   coefficients rotated by M(T @ asset_to_world), evaluated at the
   HUGSIM-world direction) reproduces the native world-fixed SH exactly;
   at other frames it must equal the native function with the direction
   rigidly rotated by the instance rotation delta (appearance follows the
   object). The un-compensated difference is reported as evidence of the
   world-fixed-SH approximation in the native chain, not an export error.
3. Geometry contract: support wlh vs annotation box wlh (>10% flagged for the
   matched-size ablation), ground-anchor offset, proper rigid poses.
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

import argparse
import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(_hugsim_expand('${HUGSIM_ROOT}'))
ADAPTER = BASE/'waymo23-native-import'
sys.path.insert(0, str(ADAPTER)); sys.path.insert(0, str(BASE/'integrations/sr-hugsim'))
from scene_math import quat_matrix, sh_basis  # noqa: E402
from asset_radiance import sh_rotation  # noqa: E402
from waymo_adapter import SR_TO_HUGSIM, rigid  # noqa: E402
import asset_calibration  # noqa: E402


def full_sh(dc, rest):
    return torch.cat([dc.view(-1, 1, 3), rest], dim=1)  # (N,16,3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', required=True)
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    asset = Path(args.asset).resolve()
    provenance = _hugsim_expand(json.loads((asset/'provenance.json').read_text()))
    iid = provenance['checkpoint_instance_index']
    calibration = asset_calibration.load(asset/'asset_calibration.json', asset)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    rigid_state = ckpt['models']['RigidNodes']
    mask = rigid_state['points_ids'].flatten().long() == iid
    means_c = rigid_state['_means'][mask].double()
    dc_c = rigid_state['_features_dc'][mask].double()
    rest_c = rigid_state['_features_rest'][mask].double()

    params, _ = torch.load(asset/'gs.pth', map_location='cpu', weights_only=False)
    sh_degree, means_a, dc_a, rest_a = params[0], params[1].double(), params[2].double(), params[3].double()
    assert sh_degree == 3
    support = calibration['support']
    c_a = torch.tensor([support['raw_recenter_xz'][0], 0., support['raw_recenter_xz'][1]])
    means_recentred = means_a - c_a

    traj = np.load(asset/'trajectory.npz')
    poses, visible = traj['asset_to_world'], traj['visible'].astype(bool)

    checks = {}

    # 1. Round-trip geometry on every visible frame.
    quats = F.normalize(rigid_state['instances_quats'][:, iid].double(), dim=-1)
    trans = rigid_state['instances_trans'][:, iid].double()
    worst = 0.0
    ground_err = []
    for f in np.flatnonzero(visible):
        native = (quat_matrix(quats[f:f+1])[0].double() @ means_c.T).T + trans[f].double()
        P = rigid(poses[f])
        placed = means_recentred.numpy() @ P[:3, :3].T + P[:3, 3]
        err = float(np.abs(native.numpy() - placed).max())
        worst = max(worst, err)
        ann = None  # ground proxy: native support bottom world-z
        ground_err.append(float(placed[:, 2].min() - native[:, 2].min()))
    checks['roundtrip_max_abs_error_m'] = worst
    checks['roundtrip_frames'] = int(visible.sum())
    checks['ground_anchor_bottom_z_max_abs_m'] = float(np.max(np.abs(ground_err)))
    assert worst < 1e-5, 'trajectory/axis round-trip mismatch'

    # 2. Colour chain equality at sampled frames/directions.
    rng = np.random.default_rng(0)
    f_ref = int(provenance['sh_reference_frame'])
    R_ref = quat_matrix(quats[f_ref:f_ref+1])[0].double()
    pick = torch.linspace(0, len(dc_c)-1, 256).long()
    def evalSH(coeff, dirs):
        basis = sh_basis(dirs, 3)                      # (D,16)
        return (basis[None, :, :, None]*coeff[:, None]).sum(2) + .5
    colour_ref_err = 0.0
    colour_delta_err = 0.0
    colour_naive_err = 0.0
    for f in np.flatnonzero(visible)[::13]:
        P = rigid(poses[f])
        native_rot = (SR_TO_HUGSIM @ P)[:3, :3]
        M = torch.tensor(sh_rotation(native_rot, 3), dtype=torch.float64)
        rest_h = torch.einsum('ij,njc->nic', M[1:, 1:], rest_a)
        dirs_s = F.normalize(torch.tensor(rng.normal(size=(64, 3)), dtype=torch.float64), dim=-1)
        dirs_h = dirs_s @ SR_TO_HUGSIM[:3, :3].T     # source world -> hugsim world
        c_native = evalSH(full_sh(dc_c, rest_c)[pick].float(), dirs_s.float())
        c_managed = evalSH(full_sh(dc_a.view(-1, 3), rest_h)[pick].float(), dirs_h.float())
        colour_naive_err = max(colour_naive_err, float((c_native - c_managed).abs().max()))
        # managed appearance = f_c(W_ref @ W_f^-1 d): directions rotated by W_ref @ W_f^T
        R_w = quat_matrix(quats[f:f+1])[0].double()
        dirs_delta = dirs_s @ (R_ref @ R_w.T).T
        c_delta = evalSH(full_sh(dc_c, rest_c)[pick].float(), dirs_delta.float())
        err = float((c_delta - c_managed).abs().max())
        colour_delta_err = max(colour_delta_err, err)
        if f == f_ref:
            colour_ref_err = float((c_native - c_managed).abs().max())
    checks['colour_chain_error_at_reference_frame'] = colour_ref_err
    checks['colour_chain_error_with_rotation_delta'] = colour_delta_err
    checks['colour_native_worldfixed_vs_rigid_appearance_max'] = colour_naive_err
    assert colour_ref_err < 1e-3, 'SH colour chain mismatch at reference frame'
    assert colour_delta_err < 1e-3, 'SH rigid-rotation colour chain mismatch'

    # 3. Geometry contract.
    ann = np.asarray(provenance['annotation_size_lwh_mean'])
    sup_wlh = np.asarray(support['wlh'])            # [width, length, height]
    ann_wlh = ann[[1, 0, 2]]
    rel = np.abs(sup_wlh - ann_wlh)/ann_wlh
    checks['support_wlh'] = sup_wlh.tolist()
    checks['annotation_wlh'] = ann_wlh.tolist()
    checks['relative_size_difference'] = rel.tolist()
    checks['matched_size_ablation_required'] = bool((rel > 0.10).any())
    checks['opaque_support_points'] = support['opaque_points']
    checks['bottom_y_down_positive_m'] = support['bottom_y']
    checks['poses_all_proper_rigid'] = True  # rigid() raised otherwise
    checks['world_frame'] = 'first_waymo_ego_x_forward_y_left_z_up'
    checks['diagnostic_uncertified'] = True
    checks['n1_certified'] = False
    checks['formal_evaluation_admitted'] = False

    out = asset/'validation_report.json'
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
