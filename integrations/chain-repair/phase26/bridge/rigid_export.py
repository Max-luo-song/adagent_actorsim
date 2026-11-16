"""Export one RigidNodes vehicle instance to the HUGSIM managed-asset format.

CPU-only, read-only on every input. Diagnostic engineering tooling:
nothing here is N1-certified or admitted to formal evaluation.

Coordinate contract
-------------------
Checkpoint RigidNodes canonical frame (Waymo annotation object frame):
    x = box length / heading direction, y = box width, z = up; origin = box centre.
Managed asset frame (waymo23-native-import/asset_geometry.py):
    x = longitudinal, y = down, z = lateral; wlh = [width, length, height].
Mapping A (canonical -> asset): asset_x = canonical_x, asset_y = -canonical_z,
asset_z = canonical_y; det(A) = +1 (proper rotation).

SH colour: the native chain evaluates canonical coefficients at the source-world
direction (world-fixed SH, live_env.py convention, validated error-0 against
archived renders). A managed asset's coefficients instead rotate rigidly with
the asset pose. We therefore store C_asset = M(A @ W_ref^T) @ C_canonical with
W_ref the instance rotation at the first visible frame: the export reproduces
the native colours exactly at the reference pose, and appearance rotates
rigidly (physically correct) when the instance pose rotates away from it.
Verified numerically by validate_export.py.

Trajectory: per visible frame f the managed pose is
    P_f = [ W_f @ A^T | t_f + (W_f @ A^T) @ c_a ]
where W_f/t_f are the (normalised, pose-refined) instance rotation/translation
in first-ego source world and c_a = [recenter_x, 0, recenter_z] is the support
recentering that NativeSceneBackend.apply subtracts from gs.pth means.
Placed points then coincide with the native rigid_geometry world points.
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
import hashlib
import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(_hugsim_expand('${HUGSIM_ROOT}'))
ADAPTER = BASE/'waymo23-native-import'
sys.path.insert(0, str(ADAPTER))
from scene_math import quat_matrix, matrix_quat, quat_mul  # noqa: E402
from asset_radiance import sh_rotation  # noqa: E402
import asset_calibration  # noqa: E402

CONVERTER_VERSION = 'sr_rigid_to_managed_v1'
# canonical (x=len, y=width, z=up) -> asset (x=longitudinal, y=down, z=lateral)
CANONICAL_TO_ASSET = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
RIGIDNODE_SEMANTIC_INDEX = 13  # live_env.names['RigidNodes']
NUM_SEMANTIC_CLASSES = 20


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def load_rigid_nodes(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'RigidNodes' not in ckpt['models']:
        raise ValueError('Checkpoint has no RigidNodes model')
    return ckpt


def identity_map(inputs, rigid_state):
    """Match checkpoint instance order to dataset instances via size + visibility."""
    info = _hugsim_expand(json.loads((Path(inputs)/'instances/instances_info.json').read_text()))
    sizes = rigid_state['instances_size'].numpy()
    fv = rigid_state['instances_fv'].numpy().astype(bool)
    mapping = {}
    used = set()
    for j in range(len(sizes)):
        best = None
        for k, v in info.items():
            if k in used or v['class_name'] != 'Vehicle':
                continue
            fa = v['frame_annotations']
            ann = np.asarray(fa['box_size'], dtype=float).mean(0)
            if np.allclose(ann, sizes[j], atol=0.05) and len(fa['frame_idx']) == int(fv[:, j].sum()):
                best = k
                break
        if best is None:
            raise ValueError(f'No unique dataset identity for checkpoint instance {j}')
        mapping[j] = best
        used.add(best)
    return mapping, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True, help='archived scene inputs (instances/, ego_pose/, ...)')
    p.add_argument('--instance', type=int, required=True, help='RigidNodes model instance index')
    p.add_argument('--output', required=True, help='new asset directory (must not exist)')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Asset directory is immutable; use a new versioned path: '+str(output))
    output.mkdir(parents=True)

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = load_rigid_nodes(ckpt_path)
    rigid = ckpt['models']['RigidNodes']
    iid = int(args.instance)
    frames, num_instances = rigid['instances_fv'].shape
    if not 0 <= iid < num_instances:
        raise ValueError('Instance index out of range')
    mapping, info = identity_map(args.inputs, rigid)
    dataset_id = mapping[iid]
    if info[dataset_id]['class_name'] != 'Vehicle':
        raise ValueError('Refusing to export a non-vehicle instance as a vehicle asset')

    mask = rigid['points_ids'].flatten().long() == iid
    n = int(mask.sum())
    if n < 1000:
        raise ValueError('Instance has too few Gaussians: '+str(n))
    means_c = rigid['_means'][mask].double()
    size = rigid['instances_size'][iid].double().numpy()  # [length, width, height]
    over = (means_c.abs() > torch.tensor(size)/2 + 0.25).any(-1)
    if over.float().mean() > 0.01:
        raise ValueError('Canonical points exceed the annotation box by >25cm for >1% of points')

    A = torch.tensor(CANONICAL_TO_ASSET, dtype=torch.float64)
    q_a = matrix_quat(A.float()).double()  # (4,) wxyz
    means_a = means_c @ A.T
    quats_a = quat_mul(q_a.expand(n, 4), F.normalize(rigid['_quats'][mask].double(), dim=-1))
    quats_a = F.normalize(quats_a, dim=-1)
    dc = rigid['_features_dc'][mask].double()          # (N,3)
    rest_c = rigid['_features_rest'][mask].double()    # (N,15,3)
    quats_all = F.normalize(rigid['instances_quats'][:, iid].double(), dim=-1)
    visible0 = rigid['instances_fv'][:, iid].bool().numpy()
    f_ref = int(np.flatnonzero(visible0)[0])
    W_ref = quat_matrix(quats_all[f_ref:f_ref+1]).double()[0].numpy()
    sh_ref = CANONICAL_TO_ASSET @ W_ref.T
    M = torch.tensor(sh_rotation(sh_ref, 3), dtype=torch.float64)
    rest_a = torch.einsum('ij,njc->nic', M[1:, 1:], rest_c).contiguous()
    scales = rigid['_scales'][mask].double()
    opac = rigid['_opacities'][mask].double().reshape(n, 1)
    logits = torch.full((n, NUM_SEMANTIC_CLASSES), -12., dtype=torch.float64)
    logits[:, RIGIDNODE_SEMANTIC_INDEX] = 12.

    params = [3,
              means_a.float(),
              dc.view(n, 1, 3).float(),
              rest_a.float(),
              logits.float(),
              scales.float(),
              quats_a.float(),
              opac.float(),
              None]
    torch.save((params, int(ckpt.get('step', 0))), output/'gs.pth')

    annotation = np.asarray(info[dataset_id]['frame_annotations']['box_size'], dtype=float)
    ann_lwh = annotation.mean(0)
    ann_lwh_std = annotation.std(0)
    wlh = [float(ann_lwh[1]), float(ann_lwh[0]), float(ann_lwh[2])]  # [width, length, height]
    (output/'wlh.json').write_text(json.dumps(wlh))

    calibration = asset_calibration.create(output, output/'asset_calibration.json')
    support = calibration['support']
    c_a = np.array([support['raw_recenter_xz'][0], 0., support['raw_recenter_xz'][1]])

    # Per-frame managed poses from the pose-refined instance trajectory.
    quats = quats_all
    trans = rigid['instances_trans'][:, iid].double()
    visible = visible0
    asset_to_world = np.zeros((frames, 4, 4))
    asset_to_world[:, 3, 3] = 1.
    heading = np.zeros(frames)
    for f in range(frames):
        R_w = quat_matrix(quats[f:f+1]).double()[0].numpy()
        t_w = trans[f].numpy()
        R_p = R_w @ CANONICAL_TO_ASSET.T
        asset_to_world[f, :3, :3] = R_p
        asset_to_world[f, :3, 3] = t_w + R_p @ c_a
        heading[f] = np.arctan2(R_w[1, 0], R_w[0, 0])
    np.savez_compressed(output/'trajectory.npz',
                        asset_to_world=asset_to_world.astype(np.float64),
                        visible=visible, heading_source_radians=heading,
                        fps=10., frame_period_s=0.1)

    provenance = dict(
        schema='sr_rigid_export_provenance_v1', converter=CONVERTER_VERSION,
        diagnostic_uncertified=True, n1_certified=False, formal_evaluation_admitted=False,
        source_checkpoint=dict(path=str(ckpt_path), **fingerprint(ckpt_path)),
        source_inputs=str(Path(args.inputs).resolve()),
        checkpoint_instance_index=iid, dataset_instance_id=dataset_id,
        dataset_class=info[dataset_id]['class_name'],
        gaussian_points=n, frames=frames, visible_frames=int(visible.sum()),
        canonical_frame='waymo_object_box_centre: x=length/heading, y=width, z=up',
        asset_frame='x=longitudinal, y=down, z=lateral (asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1)',
        canonical_to_asset=CANONICAL_TO_ASSET.tolist(),
        sh_rest_rotated_by='M(canonical_to_asset @ W_ref.T) degree blocks 1..3 via asset_radiance.sh_rotation',
        sh_reference_frame=f_ref,
        annotation_size_lwh_mean=ann_lwh.tolist(), annotation_size_lwh_std=ann_lwh_std.tolist(),
        annotation_wlh_written=wlh, gaussian_support_wlh=support['wlh'],
        trajectory_pose='pose-refined instances_quats/instances_trans; P_f=W_f@A.T with recenter offset',
        colour_chain='native sh_eval(coeff_canonical, source_world_dir); managed rotates rest by M(T@asset_to_world)',
        notes='PLY-style merged exports were not used; per-instance Gaussian tuple written directly')
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(asset=str(output), dataset_id=dataset_id, points=n,
                          support_wlh=support['wlh'], annotation_wlh=wlh,
                          visible_frames=int(visible.sum())), indent=2))


if __name__ == '__main__':
    main()
