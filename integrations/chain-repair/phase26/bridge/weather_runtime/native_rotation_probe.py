"""Pre-run native edit with a same-model, same-pose clear-render comparison."""

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
pass  # Path is provided by the portable boundary above

import numpy as np
from PIL import Image

from runtime_contract import CAMERAS


def probe_native_rotation(backend, edit, output, render_clear, apply_edit=None):
    from native_edit import rotate_native
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    spec = backend.spec
    inputs = Path(spec.source['inputs'])
    start = spec.source['start_timestep']
    pose = np.linalg.inv(np.loadtxt(inputs / 'ego_pose' / f'{start:03d}.txt')) @ np.loadtxt(
        inputs / 'ego_pose' / f'{start + 80:03d}.txt')
    report = dict(passed=False, formal_evaluation_admitted=False,
                  diagnostic_uncertified=True, scope='pre-run rigid root world-Z rotation about each frame center; same loaded model',
                  source_frame=80, ego_to_world=pose.tolist())
    try:
        # This is render-only, before control or UniAD history exists. Reset the
        # runtime weather epoch afterward; no probe image enters weather history.
        before_view = render_clear(backend, pose, 8.)
        getattr(backend, 'clear_observation_history', backend.reset)()
        model = backend.trainer.models[edit['group']]
        trans = model.instances_trans.detach().clone()
        other_groups = {(name, key): getattr(node, key).detach().clone()
                        for name, node in backend.trainer.models.items() if name != edit['group']
                        for key in ('instances_trans', 'instances_quats', 'smpl_qauts') if hasattr(node, key)}
        body = {key: getattr(model, key).detach().clone()
                for key in ('instances_quats', 'smpl_qauts') if hasattr(model, key)}
        local = {key: getattr(model, key).detach().clone()
                 for key in ('_means', '_quats', '_scales', '_opacities', '_features_dc', 'point_ids')
                 if hasattr(model, key)}
        from native_radiance import radiance_state
        radiance_before = radiance_state(model, edit['instance'])
        network = {key: value.detach().clone() for key, value in model.deform_network.state_dict().items()
                   } if hasattr(model, 'deform_network') else {}
        report['edit'] = apply_edit(edit) if apply_edit is not None else rotate_native(backend, edit)
        from scipy.spatial.transform import Rotation
        before_q = body['instances_quats'].cpu().numpy()
        after_q = model.instances_quats.detach().cpu().numpy()
        idx = edit['instance']
        expected = Rotation.from_euler('z', edit['yaw_degrees'], degrees=True).as_matrix() @ Rotation.from_quat(
            before_q[:, idx][:, [1, 2, 3, 0]]).as_matrix()
        actual = Rotation.from_quat(after_q[:, idx][:, [1, 2, 3, 0]]).as_matrix()
        report['rotation_matrices_exact'] = bool(np.allclose(expected, actual, atol=1e-6, rtol=0))
        report['all_root_positions_unchanged'] = bool(backend.torch.equal(trans, model.instances_trans))
        report['other_instance_rotations_unchanged'] = bool(np.array_equal(
            np.delete(before_q, idx, axis=1), np.delete(after_q, idx, axis=1)))
        report['proxy_tensor_shared'] = all(backend.env.models[edit['group']][key] is getattr(model, key)
            for key in ('instances_trans', 'instances_quats'))
        report['unit_quaternions'] = bool(np.allclose(np.linalg.norm(after_q[:, idx], axis=-1), 1, atol=1e-6))
        report['local_gaussian_geometry_unchanged'] = bool(local) and all(backend.torch.equal(
            value, getattr(model, key).detach()) for key, value in local.items())
        # Independent SH evaluation checks the direction relation, not just M.
        if radiance_before is not None:
            from asset_radiance import evaluator
            from scipy.spatial.transform import Rotation
            ids, before_rest, degree = radiance_before
            select = np.linspace(0, len(ids)-1, min(32, len(ids))).astype(int)
            offset_ids = ids.new_tensor(select)
            sample_ids = ids[offset_ids]
            dc = model._features_dc[sample_ids].detach().cpu().numpy()[:, None, :]
            old = np.concatenate((dc, before_rest[offset_ids].cpu().numpy()), axis=1)
            new = np.concatenate((dc, model._features_rest[sample_ids].detach().cpu().numpy()), axis=1)
            rng = np.random.default_rng(42)
            directions = rng.normal(size=(64, 3)); directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            delta = Rotation.from_euler('z', edit['yaw_degrees'], degrees=True).as_matrix()
            expected_rgb = evaluator()(degree, old.transpose(0,2,1)[:,None], directions[None] @ delta)
            actual_rgb = evaluator()(degree, new.transpose(0,2,1)[:,None], directions[None])
            report['sh_edit_direction_max_abs'] = float(np.max(np.abs(actual_rgb-expected_rgb)))
            report['sh_edit_direction_consistent'] = report['sh_edit_direction_max_abs'] < 2e-5
        else:
            report['sh_edit_direction_consistent'] = True  # Geometry-only CPU fixtures.
        # Independently reconstruct the real evaluator's oriented box at frame80.
        from native_actor_evaluation import NativeActorEvaluation
        state = backend.env.models[edit['group']]
        box = next(row for row in NativeActorEvaluation({edit['group']: state}).geometry(80.) if row[1] == idx)
        size = state['instances_size'][idx].detach().cpu().numpy()
        corners = np.array([[x*size[0]/2, y*size[1]/2, z*size[2]/2]
                            for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        world = corners @ expected[80].T + trans[80, idx].cpu().numpy()
        from scipy.spatial import ConvexHull
        polygon = world[ConvexHull(world[:, :2]).vertices, :2]
        # Hull order can change with tiny floating-point perturbations.
        distances = np.linalg.norm(polygon[:, None] - box[2][None, :], axis=-1)
        report['oriented_proxy_matches_rotation'] = bool(
            polygon.shape == box[2].shape and np.max(distances.min(axis=1)) < 1e-5
            and abs(world[:, 2].min()-box[3]) < 1e-5 and abs(world[:, 2].max()-box[4]) < 1e-5)
        report['proxy_polygon_frame80'] = box[2].tolist()
        report['deformation_network_unchanged'] = all(backend.torch.equal(
            value, model.deform_network.state_dict()[key]) for key, value in network.items())
        report['other_groups_unchanged'] = all(backend.torch.equal(
            value, getattr(backend.trainer.models[name], key).detach())
            for (name, key), value in other_groups.items())
        report['group'] = edit['group']
        report['instance'] = edit['instance']
        after_view = render_clear(backend, pose, 8.)
        report['cameras'] = {}
        for name in CAMERAS:
            a, b = before_view['cameras'][name], after_view['cameras'][name]
            for label, view in (('before', a), ('after', b)):
                Image.fromarray(view['rgb']).save(folder / f'{name}_{label}.png')
                Image.fromarray(view['weather_dynamic_mask']).save(folder / f'{name}_{label}_mask.png')
            report['cameras'][name] = dict(
                rgb_changed_pixels=int(np.any(a['rgb'] != b['rgb'], -1).sum()),
                mask_changed_pixels=int(np.count_nonzero(a['weather_dynamic_mask'] != b['weather_dynamic_mask'])),
                camera_unchanged=bool(np.array_equal(a['camera_to_world'], b['camera_to_world'])
                                      and np.array_equal(a['K'], b['K'])))
        report['passed'] = (all(report[key] for key in (
            'rotation_matrices_exact', 'all_root_positions_unchanged', 'other_instance_rotations_unchanged',
            'unit_quaternions', 'local_gaussian_geometry_unchanged', 'sh_edit_direction_consistent', 'oriented_proxy_matches_rotation', 'proxy_tensor_shared',
            'deformation_network_unchanged', 'other_groups_unchanged')) and
            all(c['camera_unchanged'] for c in report['cameras'].values()) and
            any(c['rgb_changed_pixels'] > 0 and c['mask_changed_pixels'] > 0
                for c in report['cameras'].values()))
        if not report['passed']:
            raise RuntimeError('native edit probe failed; inspect native_rotation_probe/report.json')
        return report
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Retain edited geometry; discard only render/weather observation history.
        try:
            getattr(backend, 'clear_observation_history', backend.reset)()
        finally:
            (folder / 'report.json').write_text(json.dumps(report, indent=2))
