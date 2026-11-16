"""Config-driven general RigidNodes export entry (diagnostic, uncertified).

Generalises the validated phase26 export logic (bridge/rigid_export.py, kept
frozen as the provenance of the running arms) to any scene/instance via a JSON
config. No scene-number branches: scene identity enters only through the
checkpoint/inputs paths. Source hashes, coordinate definitions and size
provenance are preserved verbatim.

Config JSON:
{
  "checkpoint": "<scene checkpoint_final.pth>",
  "inputs": "<archived scene inputs dir (instances/, ego_pose/, ...)>",
  "instance": {"model_index": 2}            // or {"dataset_id": "25"}
  "reference_frame": "first_visible",       // or an integer frame index
  "output": "<new asset dir; must not exist>",
  "note": "optional free-text note"
}
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

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))  # bridge/ (rigid_export lives there)
from rigid_export import (CANONICAL_TO_ASSET, RIGIDNODE_SEMANTIC_INDEX,
                          NUM_SEMANTIC_CLASSES, fingerprint, load_rigid_nodes,
                          identity_map)  # noqa: E402
from scene_math import quat_matrix, matrix_quat, quat_mul  # noqa: E402
from asset_radiance import sh_rotation  # noqa: E402
import asset_calibration  # noqa: E402

CONVERTER_VERSION = 'sr_rigid_to_managed_config_v1'


def resolve_instance(rigid, mapping, info, spec):
    if set(spec) == {'model_index'}:
        iid = int(spec['model_index'])
        if not 0 <= iid < rigid['instances_fv'].shape[1]:
            raise ValueError('model_index out of range')
        return iid
    if set(spec) == {'dataset_id'}:
        target = str(spec['dataset_id'])
        hits = [j for j, k in mapping.items() if k == target]
        if len(hits) != 1:
            raise ValueError(f'dataset_id {target} maps to {len(hits)} model instances')
        return hits[0]
    raise ValueError('instance must specify exactly one of model_index / dataset_id')


def export_from_config(config_path):
    cfg = _hugsim_expand(json.loads(Path(config_path).read_text()))
    allowed = {'checkpoint', 'inputs', 'instance', 'reference_frame', 'output', 'note'}
    if set(cfg)-allowed or not {'checkpoint', 'inputs', 'instance', 'output'} <= set(cfg):
        raise ValueError('Invalid export config fields')
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError('Asset directory is immutable; use a new versioned path: '+str(output))

    ckpt_path = Path(cfg['checkpoint']).resolve()
    ckpt = load_rigid_nodes(ckpt_path)
    rigid = ckpt['models']['RigidNodes']
    mapping, info = identity_map(cfg['inputs'], rigid)
    iid = resolve_instance(rigid, mapping, info, cfg['instance'])
    dataset_id = mapping[iid]
    if info[dataset_id]['class_name'] != 'Vehicle':
        raise ValueError('Refusing to export a non-vehicle instance as a vehicle asset')

    mask = rigid['points_ids'].flatten().long() == iid
    n = int(mask.sum())
    if n < 1000:
        raise ValueError('Instance has too few Gaussians: '+str(n))
    means_c = rigid['_means'][mask].double()
    size = rigid['instances_size'][iid].double().numpy()
    over = (means_c.abs() > torch.tensor(size)/2+0.25).any(-1)
    if over.float().mean() > 0.01:
        raise ValueError('Canonical points exceed the annotation box by >25cm for >1% of points')

    quats_all = F.normalize(rigid['instances_quats'][:, iid].double(), dim=-1)
    visible = rigid['instances_fv'][:, iid].bool().numpy()
    ref = cfg.get('reference_frame', 'first_visible')
    f_ref = int(np.flatnonzero(visible)[0]) if ref == 'first_visible' else int(ref)
    if not visible[f_ref]:
        raise ValueError('reference_frame not visible for this instance')

    A = torch.tensor(CANONICAL_TO_ASSET, dtype=torch.float64)
    q_a = matrix_quat(A.float()).double()
    means_a = means_c @ A.T
    quats_a = F.normalize(quat_mul(q_a.expand(n, 4), F.normalize(rigid['_quats'][mask].double(), dim=-1)), dim=-1)
    dc = rigid['_features_dc'][mask].double()
    rest_c = rigid['_features_rest'][mask].double()
    W_ref = quat_matrix(quats_all[f_ref:f_ref+1]).double()[0].numpy()
    M = torch.tensor(sh_rotation(CANONICAL_TO_ASSET @ W_ref.T, 3), dtype=torch.float64)
    rest_a = torch.einsum('ij,njc->nic', M[1:, 1:], rest_c).contiguous()
    scales = rigid['_scales'][mask].double()
    opac = rigid['_opacities'][mask].double().reshape(n, 1)
    logits = torch.full((n, NUM_SEMANTIC_CLASSES), -12., dtype=torch.float64)
    logits[:, RIGIDNODE_SEMANTIC_INDEX] = 12.

    output.mkdir(parents=True)
    params = [3, means_a.float(), dc.view(n, 1, 3).float(), rest_a.float(),
              logits.float(), scales.float(), quats_a.float(), opac.float(), None]
    torch.save((params, int(ckpt.get('step', 0))), output/'gs.pth')

    annotation = np.asarray(info[dataset_id]['frame_annotations']['box_size'], dtype=float)
    ann_lwh = annotation.mean(0)
    wlh = [float(ann_lwh[1]), float(ann_lwh[0]), float(ann_lwh[2])]
    (output/'wlh.json').write_text(json.dumps(wlh))

    calibration = asset_calibration.create(output, output/'asset_calibration.json')
    support = calibration['support']
    c_a = np.array([support['raw_recenter_xz'][0], 0., support['raw_recenter_xz'][1]])

    frames = rigid['instances_fv'].shape[0]
    trans = rigid['instances_trans'][:, iid].double()
    asset_to_world = np.zeros((frames, 4, 4)); asset_to_world[:, 3, 3] = 1.
    heading = np.zeros(frames)
    for f in range(frames):
        R_w = quat_matrix(quats_all[f:f+1]).double()[0].numpy()
        R_p = R_w @ CANONICAL_TO_ASSET.T
        asset_to_world[f, :3, :3] = R_p
        asset_to_world[f, :3, 3] = trans[f].numpy() + R_p @ c_a
        heading[f] = np.arctan2(R_w[1, 0], R_w[0, 0])
    np.savez_compressed(output/'trajectory.npz', asset_to_world=asset_to_world,
                        visible=visible, heading_source_radians=heading,
                        fps=10., frame_period_s=0.1)

    provenance = dict(
        schema='sr_rigid_export_provenance_v1', converter=CONVERTER_VERSION,
        diagnostic_uncertified=True, n1_certified=False, formal_evaluation_admitted=False,
        source_checkpoint=dict(path=str(ckpt_path), **fingerprint(ckpt_path)),
        source_inputs=str(Path(cfg['inputs']).resolve()),
        checkpoint_instance_index=iid, dataset_instance_id=dataset_id,
        dataset_class=info[dataset_id]['class_name'],
        gaussian_points=n, frames=frames, visible_frames=int(visible.sum()),
        canonical_frame='waymo_object_box_centre: x=length/heading, y=width, z=up',
        asset_frame='x=longitudinal, y=down, z=lateral (asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1)',
        canonical_to_asset=CANONICAL_TO_ASSET.tolist(),
        sh_rest_rotated_by='M(canonical_to_asset @ W_ref.T) degree blocks 1..3 via asset_radiance.sh_rotation',
        sh_reference_frame=f_ref, reference_frame_spec=cfg.get('reference_frame', 'first_visible'),
        annotation_size_lwh_mean=ann_lwh.tolist(), annotation_size_lwh_std=annotation.std(0).tolist(),
        annotation_wlh_written=wlh, gaussian_support_wlh=support['wlh'],
        trajectory_pose='pose-refined instances_quats/instances_trans; P_f=W_f@A.T with recenter offset',
        note=cfg.get('note'))
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(asset=str(output), dataset_id=dataset_id, points=n,
                          support_wlh=support['wlh'], annotation_wlh=wlh,
                          reference_frame=f_ref, visible_frames=int(visible.sum())), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    export_from_config(p.parse_args().config)
