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


def verify_translation(torch, before, after, instance, offset):
    expected = before.clone()
    expected[:, instance] += expected.new_tensor(offset)
    return bool(torch.allclose(after, expected, atol=1e-6, rtol=0))


def probe_native_edit(backend, edit, output, render_clear, apply_edit=None):
    from native_edit import translate_native
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    spec = backend.spec
    inputs = Path(spec.source['inputs'])
    start = spec.source['start_timestep']
    pose = np.linalg.inv(np.loadtxt(inputs / 'ego_pose' / f'{start:03d}.txt')) @ np.loadtxt(
        inputs / 'ego_pose' / f'{start + 80:03d}.txt')
    report = dict(passed=False, formal_evaluation_admitted=False,
                  diagnostic_uncertified=True, scope='pre-run native root translation; same loaded model',
                  source_frame=80, ego_to_world=pose.tolist())
    try:
        # This is render-only, before control or UniAD history exists. Reset the
        # runtime weather epoch afterward; no probe image enters weather history.
        before_view = render_clear(backend, pose, 8.)
        getattr(backend, 'clear_observation_history', backend.reset)()
        model = backend.trainer.models[edit['group']]
        trans = model.instances_trans.detach().clone()
        other_groups = {name: node.instances_trans.detach().clone()
                        for name, node in backend.trainer.models.items()
                        if name != edit['group'] and hasattr(node, 'instances_trans')}
        body = {key: getattr(model, key).detach().clone()
                for key in ('instances_quats', 'smpl_qauts') if hasattr(model, key)}
        network = {key: value.detach().clone() for key, value in model.deform_network.state_dict().items()
                   } if hasattr(model, 'deform_network') else {}
        report['edit'] = apply_edit(edit) if apply_edit is not None else translate_native(backend, edit)
        report['translation_exact'] = verify_translation(backend.torch, trans,
            model.instances_trans.detach(), edit['instance'], edit['offset_world_m'])
        report['proxy_tensor_shared'] = (
            backend.env.models[edit['group']]['instances_trans'] is model.instances_trans)
        report['body_parameters_unchanged'] = bool(body) and all(backend.torch.equal(
            value, getattr(model, key).detach()) for key, value in body.items())
        report['deformation_network_unchanged'] = all(backend.torch.equal(
            value, model.deform_network.state_dict()[key]) for key, value in network.items())
        report['other_groups_unchanged'] = all(backend.torch.equal(
            value, backend.trainer.models[name].instances_trans.detach())
            for name, value in other_groups.items())
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
            'translation_exact', 'proxy_tensor_shared', 'body_parameters_unchanged',
            'deformation_network_unchanged', 'other_groups_unchanged')) and
            all(c['camera_unchanged'] for c in report['cameras'].values()) and
            any(c['rgb_changed_pixels'] > 0 and c['mask_changed_pixels'] > 0
                for c in report['cameras'].values()))
        if not report['passed']:
            raise RuntimeError('native edit probe failed; inspect native_edit_probe/report.json')
        return report
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Retain edited geometry; discard only render/weather observation history.
        getattr(backend, 'clear_observation_history', backend.reset)()
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
