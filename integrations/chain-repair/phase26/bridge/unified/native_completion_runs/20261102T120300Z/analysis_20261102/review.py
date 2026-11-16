"""CPU-only independent review of completed native-edit GPU run artifacts."""

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

import hashlib
import json
pass  # Path is provided by the portable boundary above

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
CAMS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return _hugsim_expand(json.loads(path.read_text()))


def pixels(path, rgb=True):
    with Image.open(path) as im:
        return np.array(im.convert('RGB') if rgb else im)


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    originals = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*')
                 if p.is_file() and OUT not in p.parents and p.suffix in ('.json', '.png')}
    acceptance = load(RUN / 'acceptance.json')
    checks = {'top_level_passed': acceptance['passed'] is True}
    arms, views, implementations, histories = {}, [], {}, []
    implementations.update(load(RUN / 'implementation.json'))
    for kind in ('rigid', 'deformable'):
        root = RUN / kind
        data = load(root / 'acceptance.json')
        probe = load(root / 'native_edit_probe/report.json')
        weather = load(root / 'weather/report.json')['records']
        receipts = load(root / 'weather_input_receipts.json')
        packets = load(root / 'weather_packet_audit.json')
        report = load(root / 'closed_loop/run_report.json')
        reset = load(root / 'unified_reset_audit.json')
        journal = load(root / 'unified_edit_journal.json')
        implementations.update(load(root / 'implementation.json'))
        traces = report['trace']
        observations = report['observations']
        histories.append([t['temporal'] for t in traces])
        cases = {}
        for cam in CAMS:
            folder = root / 'native_edit_probe'
            cases[cam] = dict(
                rgb_changed_pixels=int(np.any(pixels(folder / f'{cam}_before.png') != pixels(folder / f'{cam}_after.png'), axis=-1).sum()),
                mask_changed_pixels=int(np.count_nonzero(pixels(folder / f'{cam}_before_mask.png', False) != pixels(folder / f'{cam}_after_mask.png', False))))
        for i, record in enumerate(weather):
            for cam in CAMS:
                folder = root / 'weather' / f'step_{i:03d}'
                image = pixels(folder / f'{cam}_weather.png')
                clear = pixels(folder / f'{cam}_clear.png')
                mask = pixels(folder / f'{cam}_mask.png', False) > 0
                h = digest(image)
                receipt = receipts.get(f'step_{i:03d}')
                views.append(dict(arm=kind, step=i, camera=cam,
                    weather_hash=h == record['cameras'][cam]['rgb_sha256'],
                    clear_hash=digest(clear) == record['cameras'][cam]['clear_sha256'],
                    harness_matches=bool(np.array_equal(image, pixels(root / 'closed_loop/closed_loop' / cam / f'{i:03d}.png'))),
                    dynamic_preserved=bool(np.array_equal(image[mask], clear[mask])),
                    packet_matches=h == packets[i]['rgb_sha256'][cam],
                    submitted_matches=None if receipt is None else (
                        h == receipt['rgb_sha256'][cam] and receipt['timestamp'] == record['timestamp'])))
        arm_views = [v for v in views if v['arm'] == kind]
        checks.update({kind + '_' + k: bool(v) for k, v in dict(
            acceptance_20=len(data['checks']) == 20 and data['passed'] and all(data['checks'].values()) and data == acceptance[kind],
            probe_state=probe['passed'] and all(probe[k] for k in ['translation_exact', 'proxy_tensor_shared', 'body_parameters_unchanged', 'deformation_network_unchanged', 'other_groups_unchanged']),
            probe_pixels=all(all(cases[cam][k] == probe['cameras'][cam][k] for k in cases[cam]) for cam in CAMS),
            visible_translation=any(c['rgb_changed_pixels'] > 0 and c['mask_changed_pixels'] > 0 for c in cases.values()),
            six_observations_five_controls=len(weather) == len(observations) == len(packets) == 6 and len(traces) == 5,
            eighteen_views=len(arm_views) == 18 and all(all(v[k] for k in ['weather_hash', 'clear_hash', 'harness_matches', 'dynamic_preserved', 'packet_matches']) for v in arm_views),
            fifteen_inputs=len(receipts) == 5 and sum(v['submitted_matches'] is True for v in arm_views) == 15,
            five_inferences=all(load(root / f'closed_loop/uniad_worker/done_{i:03d}.json')['status'] == 'ok' for i in range(5)),
            pose_and_clock=all(np.array_equal(r['ego_to_world'], o['closed_ego_to_world']) and r['timestamp'] == o['timestamp'] for r,o in zip(weather, observations)),
            camera_mounts=all(np.allclose(np.array(r['ego_to_world']) @ np.array(c['camera_to_ego']), c['camera_to_world'], atol=1e-8, rtol=0) for r in weather for c in r['cameras'].values()),
            causal_windows=[r['audit']['window_frames'] for r in weather] == [[80,80,80],[80,80,81],[80,81,82],[81,82,83],[82,83,84],[83,84,85]] and all(not r['audit']['future_frames_used'] for r in weather),
            single_edit_revision=len(journal) == 1 and journal[0]['command'] == data['unified_edit_command'] and all(p['edited'] and p['edit_revision'] == 1 for p in packets),
            reset=len(reset) == 1 and all(reset[0].values()),
            original_implementation_stable=report['implementation_unchanged'] and report['checkpoint_unchanged'],
            diagnostic_only=data['formal_evaluation_admitted'] is False).items()})
        positions = np.array([o['closed_ego_to_world'] for o in observations])[:, :3, 3]
        arms[kind] = dict(probe_cases=cases, reset=reset, metrics=dict(
            simulation_seconds=observations[-1]['timestamp'] - observations[0]['timestamp'],
            path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            initial_speed_mps=traces[0]['state_before'][3], final_speed_mps=traces[-1]['state_after'][3],
            joint_wall_seconds=report['wall_seconds'], minimum_actor_gap_m=report['minimum_all_actor_reference_gap_m'],
            minimum_predicted_road_clearance_m=min(t['predicted_road_swept_clearance_m'] for t in traces)))
    lifecycle = load(RUN / 'lifecycle/probe/report.json')
    checks['lifecycle_31_recorded_checks'] = (len(lifecycle['checks']) == 31 and lifecycle['passed'] and all(lifecycle['checks'].values()) and lifecycle == acceptance['lifecycle'])
    lc_views = []
    for i, episode in enumerate(lifecycle['episodes']):
        weather = load(RUN / 'lifecycle/weather_load_0/report.json')['records']
        for j, hashes in enumerate(episode['packet_hashes']):
            record = weather[i * 2 + j]
            for cam, expected in hashes.items():
                image = pixels(RUN / f'lifecycle/weather_load_0/step_{i * 2 + j:03d}/{cam}_weather.png')
                lc_views.append(digest(image) == expected == record['cameras'][cam]['rgb_sha256'])
    checks['lifecycle_12_packet_hashes'] = len(lc_views) == 12 and all(lc_views)
    lc_weather = load(RUN / 'lifecycle/weather_load_0/report.json')['records']
    checks['lifecycle_real_causal_history_reset'] = (
        [r['audit']['window_frames'] for r in lc_weather] == [[80,80,80],[80,80,81],[80,80,80],[80,80,81]]
        and lc_weather[0]['audit']['epoch'] != lc_weather[2]['audit']['epoch'])
    reload_weather = load(RUN / 'lifecycle/weather_load_1/report.json')['records']
    checks['reload_fresh_history'] = len(reload_weather) == 1 and reload_weather[0]['audit']['window_frames'] == [80,80,80]
    checks['separate_uniad_histories'] = (histories[0][0]['scene_token'] != histories[1][0]['scene_token']
        and all(len({t['scene_token'] for t in h}) == 1 and [t['history_bev_present'] for t in h] == [False,True,True,True,True] for h in histories))
    checks['scope_boundaries_preserved'] = (not lifecycle['different_scene_verified'] and not lifecycle['driving_test'] and not acceptance['uniad_persistent_worker_reset_verified'])
    source_checks = {p: Path(p).is_file() and sha(Path(p)) == h for p,h in implementations.items()}
    checks['current_source_fingerprints'] = all(source_checks.values())
    startup = {str(p.relative_to(RUN)): load(p) for p in RUN.rglob('startup_status.json')}
    checks['four_weather_workers_ready'] = len(startup) == 4 and all(s['stage'] == 'ready' for s in startup.values())
    checks['original_evidence_preserved'] = all(sha(RUN / p) == h for p,h in originals.items())
    summary = dict(passed=all(checks.values()), checks=checks, arms=arms, views=views,
        lifecycle_reported_checks=lifecycle['checks'], lifecycle_packet_hashes_recomputed=12,
        lifecycle_restoration_evidence='Runtime tensor/pixel assertions recorded in report and source; baseline/restored images were not separately saved for offline remeasurement.',
        source_checks=source_checks, startup=startup, original_evidence_sha256=originals,
        formal_evaluation_admitted=False, different_scene_verified=False,
        uniad_persistent_worker_reset_verified=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    sheet = Image.new('RGB', (1440, 592), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, kind in enumerate(('rigid', 'deformable')):
        folder = RUN / kind / 'native_edit_probe'
        before = pixels(folder / 'CAM_FRONT_before.png')
        after = pixels(folder / 'CAM_FRONT_after.png')
        diff = np.clip(np.abs(after.astype(np.int16) - before.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        for col, (name, data) in enumerate([('before',before),('after',after),('abs diff x8',diff)]):
            im=Image.fromarray(data); im.thumbnail((480,270))
            sheet.paste(im,(col*480,row*296+24))
            draw.text((col*480+8,row*296+4),kind+' '+name,fill='black')
    sheet.save(OUT / 'native_edit_comparison.png')
    print(json.dumps(dict(passed=summary['passed'], checks=checks, arms=arms, startup=startup),indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
