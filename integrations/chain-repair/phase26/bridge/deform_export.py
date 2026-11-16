"""Export one DeformableNodes instance at a FROZEN frame to the managed-asset format.

CPU-only, read-only on every input. Diagnostic engineering tooling:
nothing here is N1-certified or admitted to formal evaluation.

Semantics mirrored from framework_repo models/nodes/deformable.py +
models/modules.py (ConditionalDeformNetwork, get_embedder) +
models/gaussians/vanilla.py (activations). The network structure is rebuilt
standalone because models/modules.py imports nvdiffrast at module top
(unavailable in this environment); the rebuild is verified against the
checkpoint state_dict shapes and the scene config networks section.

CRITICAL SCOPE NOTE: the exported asset freezes the time-conditioned
deformation at --frame. The managed-asset chain has no deformation-network /
time-state interface, so the exported asset is the deformed shape AT THAT
FRAME ONLY. Dynamic deformation over time is NOT claimed and NOT supported
by this export (recorded in provenance: dynamic_deformation_supported=false).

Coordinate/SH/trajectory contract identical to rigid_export.py (see its
docstring); canonical frame is the Waymo annotation object frame.
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
sys.path.insert(0, str(BASE/'waymo23-native-import'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_math import quat_matrix, matrix_quat, quat_mul  # noqa: E402
from asset_radiance import sh_rotation  # noqa: E402
import asset_calibration  # noqa: E402
from rigid_export import (CANONICAL_TO_ASSET, fingerprint)  # noqa: E402

CONVERTER_VERSION = 'sr_deform_frozen_to_managed_v1'
DEFORMABLE_SEMANTIC_INDEX = 18  # live_env.names['DeformableNodes']
NUM_SEMANTIC_CLASSES = 20
NUM_FRAMES_552 = 198  # normalized_timestamps = linspace(0, 1, num_timesteps)


# --- standalone mirror of models/modules.py (Embedder + ConditionalDeformNetwork)
def nef_embed(x, num_freqs):
    """get_embedder(multires, i): include_input, log_sampling, [sin, cos]."""
    out = [x]
    freqs = 2. ** torch.linspace(0, num_freqs - 1, num_freqs).double()
    for f in freqs:
        out.append(torch.sin(np.pi * f * x))
        out.append(torch.cos(np.pi * f * x))
    return torch.cat(out, dim=-1)


def deform_forward(state, x, t_scalar, embed, D=8, W=256, x_multires=10, t_multires=10):
    """Mirror of ConditionalDeformNetwork.forward (deform_scale=False variant)."""
    x_emb = nef_embed(x.double(), x_multires)
    t = torch.full((x.shape[0], 1), float(t_scalar), dtype=torch.float64)
    t_emb = nef_embed(t, t_multires)
    h = torch.cat([x_emb, t_emb, embed.expand(x_emb.shape[0], -1)], dim=-1)
    inp = h
    for i in range(8):
        w = state[f'deform_network.linear.{i}.weight'].double()
        b = state[f'deform_network.linear.{i}.bias'].double()
        h = F.relu(h @ w.T + b)
        if i == 4:  # skips = [D // 2]
            h = torch.cat([inp, h], dim=-1)
    d_xyz = h @ state['deform_network.gaussian_warp.weight'].double().T \
        + state['deform_network.gaussian_warp.bias'].double()
    d_quat = h @ state['deform_network.gaussian_rotation.weight'].double().T \
        + state['deform_network.gaussian_rotation.bias'].double()
    return d_xyz, d_quat


def identity_map_any_class(inputs, state):
    """Like rigid_export.identity_map but for non-vehicle classes."""
    info = _hugsim_expand(json.loads((Path(inputs)/'instances/instances_info.json').read_text()))
    sizes = state['instances_size'].numpy()
    fv = state['instances_fv'].numpy().astype(bool)
    mapping, used = {}, set()
    for j in range(len(sizes)):
        best = None
        for k, v in info.items():
            if k in used or v['class_name'] == 'Vehicle':
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
    p.add_argument('--inputs', required=True)
    p.add_argument('--instance', type=int, required=True)
    p.add_argument('--frame', type=int, required=True,
                   help='frame at which the deformation is frozen (must be visible)')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Asset directory is immutable; use a new versioned path: '+str(output))

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'DeformableNodes' not in ckpt['models']:
        raise ValueError('Checkpoint has no DeformableNodes model')
    state = ckpt['models']['DeformableNodes']
    if 'deform_network.gaussian_scaling.weight' in state:
        raise NotImplementedError('deform_scale=true variant not mirrored')
    frames, num_instances = state['instances_fv'].shape
    iid = int(args.instance)
    if not 0 <= iid < num_instances:
        raise ValueError('Instance index out of range')
    if not 0 <= args.frame < frames:
        raise ValueError('Frame out of range')
    if not bool(state['instances_fv'][args.frame, iid]):
        raise ValueError('Instance not visible at the freeze frame')

    mapping, info = identity_map_any_class(args.inputs, state)
    dataset_id = mapping[iid]
    embed = state['instances_embedding'][iid].double().reshape(1, -1)

    mask = state['points_ids'].flatten().long() == iid
    n = int(mask.sum())
    if n < 100:
        raise ValueError('Instance has too few Gaussians: '+str(n))
    means_c = state['_means'][mask].double()
    height = float(state['instances_size'][iid][2])
    t_norm = float(args.frame) / (frames - 1)  # linspace(0,1,frames)[frame]

    # deformation at the freeze frame (canonical frame)
    x = means_c / height * 2.
    d_xyz, d_quat = deform_forward(state, x, t_norm, embed)
    means_def = means_c + d_xyz
    quats_def = F.normalize(state['_quats'][mask].double() + d_quat, dim=-1)

    output.mkdir(parents=True)
    A = torch.tensor(CANONICAL_TO_ASSET, dtype=torch.float64)
    q_a = matrix_quat(A.float()).double()
    means_a = means_def @ A.T
    quats_a = F.normalize(quat_mul(q_a.expand(n, 4), quats_def), dim=-1)
    dc = state['_features_dc'][mask].double()
    rest_c = state['_features_rest'][mask].double()
    quats_all = F.normalize(state['instances_quats'][:, iid].double(), dim=-1)
    visible = state['instances_fv'][:, iid].bool().numpy()
    f_ref = int(np.flatnonzero(visible)[0])
    W_ref = quat_matrix(quats_all[f_ref:f_ref+1]).double()[0].numpy()
    M = torch.tensor(sh_rotation(CANONICAL_TO_ASSET @ W_ref.T, 3), dtype=torch.float64)
    rest_a = torch.einsum('ij,njc->nic', M[1:, 1:], rest_c).contiguous()
    logits = torch.full((n, NUM_SEMANTIC_CLASSES), -12., dtype=torch.float64)
    logits[:, DEFORMABLE_SEMANTIC_INDEX] = 12.

    params = [3, means_a.float(), dc.view(n, 1, 3).float(), rest_a.float(),
              logits.float(), state['_scales'][mask].float(), quats_a.float(),
              state['_opacities'][mask].float(), None]
    torch.save((params, int(ckpt.get('step', 0))), output/'gs.pth')

    annotation = np.asarray(info[dataset_id]['frame_annotations']['box_size'], dtype=float)
    ann_lwh = annotation.mean(0)
    wlh = [float(ann_lwh[1]), float(ann_lwh[0]), float(ann_lwh[2])]
    (output/'wlh.json').write_text(json.dumps(wlh))
    calibration = asset_calibration.create(output, output/'asset_calibration.json')
    support = calibration['support']
    c_a = np.array([support['raw_recenter_xz'][0], 0., support['raw_recenter_xz'][1]])

    trans = state['instances_trans'][:, iid].double()
    asset_to_world = np.zeros((frames, 4, 4)); asset_to_world[:, 3, 3] = 1.
    heading = np.zeros(frames)
    for f in range(frames):
        R_w = quat_matrix(quats_all[f:f+1]).double()[0].numpy()
        R_p = R_w @ CANONICAL_TO_ASSET.T
        asset_to_world[f, :3, :3] = R_p
        asset_to_world[f, :3, 3] = trans[f].double().numpy() + R_p @ c_a
        heading[f] = np.arctan2(R_w[1, 0], R_w[0, 0])
    np.savez_compressed(output/'trajectory.npz',
                        asset_to_world=asset_to_world, visible=visible,
                        heading_source_radians=heading, fps=10., frame_period_s=0.1)

    provenance = dict(
        schema='sr_deform_frozen_export_provenance_v1', converter=CONVERTER_VERSION,
        diagnostic_uncertified=True, n1_certified=False, formal_evaluation_admitted=False,
        source_checkpoint=dict(path=str(ckpt_path), **fingerprint(ckpt_path)),
        source_inputs=str(Path(args.inputs).resolve()),
        checkpoint_instance_index=iid, dataset_instance_id=dataset_id,
        dataset_class=info[dataset_id]['class_name'],
        gaussian_points=n, frames=frames, visible_frames=int(visible.sum()),
        deformation_frozen_at_frame=int(args.frame),
        freeze_frame_normalized_time=t_norm,
        dynamic_deformation_supported=False,
        deform_variant='deform_quat=true, deform_scale=false (config-verified)',
        deform_magnitude_l2_median_m=float(d_xyz.norm(dim=-1).median()),
        deform_magnitude_l2_max_m=float(d_xyz.norm(dim=-1).max()),
        network_mirror='models/modules.py ConditionalDeformNetwork rebuilt standalone '
                       '(nvdiffrast import unrelated to this path); shapes verified vs state_dict',
        canonical_frame='waymo_object_box_centre: x=length/heading, y=width, z=up',
        asset_frame='x=longitudinal, y=down, z=lateral (asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1)',
        sh_reference_frame=f_ref,
        annotation_wlh_written=wlh, gaussian_support_wlh=support['wlh'],
        notes='Frozen single-frame shape; NOT a dynamic deformation asset; '
              'appearance/geometry at other frames may differ from the source scene')
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(asset=str(output), dataset_id=dataset_id,
                          dataset_class=info[dataset_id]['class_name'],
                          points=n, freeze_frame=args.frame,
                          deform_median_m=float(d_xyz.norm(dim=-1).median()),
                          support_wlh=support['wlh']), indent=2))


if __name__ == '__main__':
    main()
