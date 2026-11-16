"""Same-model insert/move/remove/reset smoke through the public edit interface."""

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

import copy
import json
pass  # Path is provided by the portable boundary above

import numpy as np
from PIL import Image

from runtime_contract import CAMERAS


def probe_managed_asset(session, command, output, render_clear):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    backend = session.backend
    start = backend.spec.source['start_timestep']
    inputs = Path(backend.spec.source['inputs'])
    pose = np.linalg.inv(np.loadtxt(inputs / 'ego_pose' / f'{start:03d}.txt')) @ np.loadtxt(
        inputs / 'ego_pose' / f'{start+80:03d}.txt')
    report = dict(passed=False, formal_evaluation_admitted=False, cases={}, receipts=[])
    arrays = {}
    def capture(name):
        value = render_clear(backend, pose, 8.)
        arrays[name] = value
        for cam in CAMERAS:
            Image.fromarray(value['cameras'][cam]['rgb']).save(folder / f'{name}_{cam}.png')
            Image.fromarray(value['cameras'][cam]['weather_dynamic_mask']).save(folder / f'{name}_{cam}_mask.png')
        backend.clear_observation_history()
    try:
        capture('baseline')
        report['receipts'].append(session.edit(command))
        capture('inserted')
        asset = backend.managed_assets.assets[command['object_id'].split(':',1)[1]]
        report['points'] = len(asset.local_means)
        report['proxy_pose_matches_command'] = bool(np.array_equal(asset.pose, np.asarray(command['asset_to_world'])))
        moved = copy.deepcopy(command['asset_to_world'])
        moved[0][3] += .5
        move = dict(schema='scene_edit_v1', mode='diagnostic_preview', operation='set_pose',
                    object_id=command['object_id'], asset_to_world=moved,
                    expected_revision=session.revision)
        report['receipts'].append(session.edit(move))
        capture('moved')
        report['receipts'].append(session.edit(dict(schema='scene_edit_v1', mode='diagnostic_preview',
            operation='remove', object_id=command['object_id'], expected_revision=session.revision)))
        capture('removed')
        for label, a, b in (('insert','baseline','inserted'),('move','inserted','moved'),('remove','baseline','removed')):
            report['cases'][label] = {cam: dict(
                rgb_changed_pixels=int(np.any(arrays[a]['cameras'][cam]['rgb'] != arrays[b]['cameras'][cam]['rgb'], -1).sum()),
                mask_changed_pixels=int(np.count_nonzero(arrays[a]['cameras'][cam]['weather_dynamic_mask'] !=
                                                        arrays[b]['cameras'][cam]['weather_dynamic_mask'])))
                for cam in CAMERAS}
        report['removed_registry_empty'] = not backend.managed_assets.assets and not session.objects
        report['passed'] = (report['proxy_pose_matches_command'] and report['removed_registry_empty'] and
            all(any(x['rgb_changed_pixels'] > 0 and x['mask_changed_pixels'] > 0
                    for x in report['cases'][label].values()) for label in ('insert','move')) and
            all(x['rgb_changed_pixels'] == x['mask_changed_pixels'] == 0 for x in report['cases']['remove'].values()))
        if not report['passed']:
            raise RuntimeError('managed asset insert/move/remove smoke failed')
        return report
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        session.reset()  # next real insert starts a fresh generation/revision/history
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
