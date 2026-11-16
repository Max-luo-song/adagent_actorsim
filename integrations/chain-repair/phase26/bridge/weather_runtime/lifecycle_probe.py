"""Real same-session reset/re-edit and load_scene checks; no driving claims."""

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
import hashlib
import json
pass  # Path is provided by the portable boundary above

import numpy as np


def rejects(call, exception):
    try:
        call()
    except exception:
        return True
    return False


def recorded_pose(spec, frame=80):
    inputs = Path(spec.source['inputs'])
    start = spec.source['start_timestep']
    return np.linalg.inv(np.loadtxt(inputs / 'ego_pose' / f'{start:03d}.txt')) @ np.loadtxt(
        inputs / 'ego_pose' / f'{start + frame:03d}.txt')


def pixel_hashes(raw):
    return {name: {key: hashlib.sha256(view[key].tobytes()).hexdigest()
                   for key in ('rgb', 'weather_dynamic_mask')}
            for name, view in raw['cameras'].items()}


def probe_lifecycle(session, manifest, commands, output, render_clear, switch_manifest=None):
    """Caller owns session/worker cleanup. Failure report is always retained."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, checks={}, episodes=[], formal_evaluation_admitted=False,
                  uniad_persistent_worker_reset_verified=False, driving_test=False,
                  different_scene_verified=False)
    checks = report['checks']
    try:
        session.load_scene(manifest)
        backend = session.backend
        first_source = copy.deepcopy(session.spec.source)
        pose = recorded_pose(session.spec)
        baseline = pixel_hashes(render_clear(backend, pose, 8.))
        backend.clear_observation_history()
        for index, original_command in enumerate(commands):
            command = dict(original_command, expected_generation=session.generation,
                           expected_revision=0)
            old_editor = session.editor
            generation = session.generation
            receipt = session.edit(command)
            packets = [session.render(pose, timestamp) for timestamp in (8., 8.1)]
            records = backend.weather_records[-2:]
            originals = dict(session.editor.originals)
            episode = dict(command=command, receipt=receipt, generation=generation,
                history_tokens=[p['history_token'] for p in packets],
                weather_audits=[r['audit'] for r in records],
                packet_hashes=[{name: hashlib.sha256(c['rgb'].tobytes()).hexdigest()
                               for name, c in p['cameras'].items()} for p in packets])
            report['episodes'].append(episode)
            checks[f'episode{index}_same_backend'] = session.backend is backend
            checks[f'episode{index}_revision'] = receipt['edit_revision'] == 1 and all(
                p['edit_revision'] == 1 and p['edit_trust']['edited'] for p in packets)
            checks[f'episode{index}_tokens'] = all(p['history_token'] == generation for p in packets)
            checks[f'episode{index}_weather_restart'] = (
                [r['audit']['window_frames'] for r in records] == [[80, 80, 80], [80, 80, 81]]
                and records[0]['audit']['epoch'] == records[1]['audit']['epoch'])
            checks[f'episode{index}_frozen_after_render'] = rejects(
                lambda: session.edit(dict(original_command, expected_revision=1)), RuntimeError)
            session.reset()
            checks[f'episode{index}_restored_positions'] = bool(originals) and all(
                backend.torch.equal(backend.trainer.models[g].instances_trans[:, i].detach(), value)
                for (g, i), value in originals.items())
            checks[f'episode{index}_reset_metadata'] = (
                session.generation != generation and session.revision == 0
                and not session.edit_journal() and not session.objects
                and not backend.managed_assets.assets and backend._last is None
                and backend.last_weather_reset_verified)
            checks[f'episode{index}_old_editor_rejected'] = rejects(old_editor.objects, RuntimeError)
            checks[f'episode{index}_stale_command_rejected'] = rejects(lambda: session.edit(command), ValueError)
            checks[f'episode{index}_rejections_do_not_commit'] = session.revision == 0 and not session.edit_journal()
            checks[f'episode{index}_restored_pixels'] = (
                pixel_hashes(render_clear(backend, pose, 8.)) == baseline)
            backend.clear_observation_history()
            if not all(checks.values()):
                raise RuntimeError('same-session episode lifecycle probe failed')
        checks['same_backend_two_episodes'] = len(report['episodes']) == 2
        checks['episode_generations_differ'] = len({e['generation'] for e in report['episodes']}) == 2
        checks['weather_epochs_differ'] = len({e['weather_audits'][0]['epoch'] for e in report['episodes']}) == 2
        old_editor, old_backend, old_generation = session.editor, session.backend, session.generation
        session.load_scene(switch_manifest or manifest)
        checks['reload_old_backend_closed'] = old_backend.closed and session.backend is not old_backend
        checks['reload_old_editor_rejected'] = rejects(old_editor.objects, RuntimeError)
        checks['reload_new_empty_generation'] = (session.generation != old_generation and
            session.revision == 0 and not session.edit_journal() and not session.objects)
        checks['reload_old_command_rejected'] = rejects(
            lambda: session.edit(command), ValueError)
        checks['reload_fresh_weather_records'] = not session.backend.weather_records
        source = session.spec.source
        distinct = any(source.get(k) != first_source.get(k) for k in ('scene', 'checkpoint', 'inputs'))
        if switch_manifest:
            checks['distinct_scene_source'] = distinct
        packet = session.render(recorded_pose(session.spec), 8.)
        record = session.backend.weather_records[-1]
        checks['reload_fresh_packet'] = (packet['history_token'] == session.generation and
            packet['edit_revision'] == 0 and not packet['edit_trust']['edited']
            and record['audit']['window_frames'] == [80, 80, 80])
        report['different_scene_verified'] = bool(switch_manifest) and distinct
        report['reload_source'] = {k: source.get(k) for k in ('scene', 'checkpoint', 'inputs')}
        report['passed'] = all(checks.values())
        if not report['passed']:
            raise RuntimeError('scene reload lifecycle probe failed')
        return report
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
