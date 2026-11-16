"""Diagnostic native-node translation. Renderer and proxy share updated tensors.

Pre-run only; no managed-asset/SMPL skeletal edit equivalence is claimed.
"""
import numpy as np


def translate_native(backend, request):
    if set(request) != {'schema','operation','group','instance','offset_world_m'}:
        raise ValueError('unexpected native edit fields')
    if request['schema'] != 'framework_native_edit_v1' or request['operation'] != 'translate':
        raise ValueError('unsupported native edit')
    group, idx = request['group'], request['instance']
    if group not in ('RigidNodes', 'SMPLNodes', 'DeformableNodes'):
        raise ValueError('only native rigid/SMPL/deformable actors supported')
    model = backend.trainer.models[group]
    if type(idx) is not int or not 0 <= idx < model.instances_trans.shape[1]:
        raise ValueError('invalid instance')
    offset = np.asarray(request['offset_world_m'], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 2:
        raise ValueError('offset must be finite 3-vector <=2m for this diagnostic')
    if backend._last is not None:
        raise RuntimeError('native translation is pre-run only')
    proxy = backend.env.models[group]['instances_trans']
    if proxy is not model.instances_trans:
        raise RuntimeError('renderer/proxy tensor identity broken')
    with backend.torch.no_grad():
        model.instances_trans[:, idx].add_(model.instances_trans.new_tensor(offset))
    return dict(request, diagnostic_uncertified=True, formal_evaluation_admitted=False,
                collision_proxy='same updated instances_trans tensor; annotation-box precision',
                body_animation_preserved=True, articulated_body=group == 'SMPLNodes')


def rotate_native(backend, request):
    """World-Z yaw about each frame's own root center, wxyz left product.

    Root positions and local Gaussian geometry remain untouched.
    World-fixed SH coefficients rotate by the editing delta, preserving radiance. The renderer
    rotates both means and covariance through its existing root transform.
    """
    from numbers import Real
    if set(request) != {'schema', 'operation', 'group', 'instance', 'yaw_degrees'}:
        raise ValueError('unexpected native rotation fields')
    if (request['schema'] != 'framework_native_edit_v1' or request['operation'] != 'rotate'
            or request['group'] != 'RigidNodes'):
        raise ValueError('native rotation supports RigidNodes only')
    yaw = request['yaw_degrees']
    if (not isinstance(yaw, Real) or isinstance(yaw, (bool, np.bool_))
            or not np.isfinite(yaw) or abs(yaw) > 45):
        raise ValueError('diagnostic yaw must be finite and <=45 degrees')
    model = backend.trainer.models['RigidNodes']
    idx = request['instance']
    if type(idx) is not int or not 0 <= idx < model.instances_trans.shape[1]:
        raise ValueError('invalid instance')
    if backend._last is not None:
        raise RuntimeError('native rotation is pre-run only')
    for key in ('instances_trans', 'instances_quats'):
        if backend.env.models['RigidNodes'][key] is not getattr(model, key):
            raise RuntimeError('renderer/proxy tensor identity broken')
    q = model.instances_quats[:, idx]
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError('expected rigid root quaternions [frames,4] in wxyz order')
    torch = backend.torch
    norm = q.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(q).all()) or not bool((norm > 1e-8).all()):
        raise ValueError('invalid source root quaternion')
    # [cos(a/2),0,0,sin(a/2)] * q. Preserve original tensor identity.
    c, s = np.cos(np.deg2rad(yaw)/2), np.sin(np.deg2rad(yaw)/2)
    w, x, y, z = (q / norm).unbind(-1)
    rotated = torch.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w), -1)
    from native_radiance import radiance_state, rotated_rest
    state = radiance_state(model, idx)
    appearance = rotated_rest(state[1], state[2], yaw) if state is not None else None
    if appearance is not None and not bool(torch.isfinite(appearance).all()):
        raise ValueError('nonfinite rotated SH coefficients')
    with torch.no_grad():
        q.copy_(rotated / rotated.norm(dim=-1, keepdim=True))
        if state is not None:
            model._features_rest[state[0]] = appearance
    return dict(request, diagnostic_uncertified=True, formal_evaluation_admitted=False,
                axis='world_z', pivot='per_frame_root_center', quaternion_order='wxyz',
                root_trajectory_preserved=True, local_gaussian_geometry_preserved=True,
                sh_rotation='world-fixed coefficients rotated by edit delta' if state is not None else 'no SH in geometry fixture',
                sh_dc_preserved=True,
                collision_proxy='same updated instances_quats tensor; annotation-box precision')
