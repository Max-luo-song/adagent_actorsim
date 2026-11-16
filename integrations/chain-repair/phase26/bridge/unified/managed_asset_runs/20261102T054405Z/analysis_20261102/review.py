"""CPU-only review of sealed managed-asset GPU artifacts; no rendering or inference."""

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
JOINT = RUN / 'joint'


def read(rel):
    return _hugsim_expand(json.loads((JOINT / rel).read_text()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def mask(path):
    with Image.open(path) as im:
        return np.array(im)


def main():
    # Hash original JSON/PNG evidence before reading; derived outputs stay separate.
    originals = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*')
                 if p.is_file() and OUT not in p.parents and p.suffix in ('.json', '.png')}
    acceptance = read('acceptance.json')
    report = read('closed_loop/run_report.json')
    weather = read('weather/report.json')['records']
    receipts = read('weather_input_receipts.json')
    packets = read('weather_packet_audit.json')
    probe = read('managed_asset_probe/report.json')
    sources = read('managed_asset_sources.json')
    cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    views = []
    for i, record in enumerate(weather):
        for cam in cameras:
            folder = JOINT / 'weather' / f'step_{i:03d}'
            image = rgb(folder / f'{cam}_weather.png')
            clear = rgb(folder / f'{cam}_clear.png')
            dynamic = mask(folder / f'{cam}_mask.png') > 0
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            submitted = receipts.get(f'step_{i:03d}')
            views.append(dict(step=i, camera=cam,
                weather_hash_matches=digest == record['cameras'][cam]['rgb_sha256'],
                clear_hash_matches=hashlib.sha256(clear.tobytes()).hexdigest() == record['cameras'][cam]['clear_sha256'],
                harness_pixels_equal=bool(np.array_equal(image, rgb(JOINT / 'closed_loop/closed_loop' / cam / f'{i:03d}.png'))),
                dynamic_pixels=int(dynamic.sum()),
                dynamic_pixels_preserved=bool(np.array_equal(image[dynamic], clear[dynamic])),
                changed_pixels=int(np.any(image != clear, axis=-1).sum()),
                packet_hash_matches=digest == packets[i]['rgb_sha256'][cam],
                input_receipt_matches=None if submitted is None else (
                    submitted['rgb_sha256'][cam] == digest and submitted['timestamp'] == record['timestamp'])) )
    cases = {}
    for label, a, b in [('insert', 'baseline', 'inserted'), ('move', 'inserted', 'moved'), ('remove', 'baseline', 'removed')]:
        cases[label] = {}
        for cam in cameras:
            folder = JOINT / 'managed_asset_probe'
            cases[label][cam] = dict(
                rgb_changed_pixels=int(np.any(rgb(folder / f'{a}_{cam}.png') != rgb(folder / f'{b}_{cam}.png'), axis=-1).sum()),
                mask_changed_pixels=int(np.count_nonzero(mask(folder / f'{a}_{cam}_mask.png') != mask(folder / f'{b}_{cam}_mask.png'))))
    implementations = _hugsim_expand(json.loads((RUN / 'implementation.json').read_text()))['files']
    implementations.update(read('implementation.json'))
    source_checks = {p: Path(p).is_file() and sha(Path(p)) == expected for p, expected in implementations.items()}
    asset_checks = {name: sha(Path(sources['asset']) / name) == desc['sha256']
                    for name, desc in sources['calibration']['source_files'].items()}
    observations = report['observations']
    traces = report['trace']
    reset = read('unified_reset_audit.json')
    journal = read('unified_edit_journal.json')
    done = [read(f'closed_loop/uniad_worker/done_{i:03d}.json') for i in range(5)]
    managed = [o['all_actor_reference']['managed_reference'] for o in observations]
    position = np.array([o['closed_ego_to_world'] for o in observations])[:, :3, 3]
    checks = dict(
        original_acceptance_20_passed=len(acceptance['checks']) == 20 and acceptance['passed'] and all(acceptance['checks'].values()),
        six_observations_five_controls=len(weather) == len(observations) == 6 and len(traces) == report['steps_executed'] == 5,
        all_18_views_match= len(views) == 18 and all(all(v[k] for k in ['weather_hash_matches', 'clear_hash_matches', 'harness_pixels_equal', 'dynamic_pixels_preserved', 'packet_hash_matches']) for v in views),
        all_15_input_receipts_match=len(receipts) == 5 and sum(v['input_receipt_matches'] is True for v in views) == 15,
        five_inferences_done=all(d.get('status') == 'ok' for d in done),
        probe_pixels_match_report=cases == probe['cases'],
        insert_and_move_visible=all(any(v['rgb_changed_pixels'] > 0 and v['mask_changed_pixels'] > 0 for v in cases[k].values()) for k in ['insert', 'move']),
        removal_restores_all_pixels=all(v['rgb_changed_pixels'] == v['mask_changed_pixels'] == 0 for v in cases['remove'].values()),
        probe_revisions=[r['edit_revision'] for r in probe['receipts']] == [1, 2, 3],
        probe_pose_and_empty_registry=probe['proxy_pose_matches_command'] and probe['removed_registry_empty'],
        six_collections_have_21396_points=all(r['managed_gaussians_in_last_collection'] == 21396 and r['managed_objects'] == ['preview_car'] for r in weather),
        six_managed_proxy_records=len(managed) == 6 and all(len(m) == 1 and m[0]['object_id'] == 'managed:preview_car' and not m[0]['overlap'] for m in managed),
        pose_feedback=all(np.array_equal(r['ego_to_world'], o['closed_ego_to_world']) for r, o in zip(weather, observations)),
        camera_mount_alignment=all(np.allclose(np.array(r['ego_to_world']) @ np.array(c['camera_to_ego']), c['camera_to_world'], atol=1e-8, rtol=0) for r in weather for c in r['cameras'].values()),
        strict_causal_windows=[r['audit']['window_frames'] for r in weather] == [[80,80,80],[80,80,81],[80,81,82],[81,82,83],[82,83,84],[83,84,85]],
        no_future_frames=all(not r['audit']['future_frames_used'] for r in weather),
        single_scene_history=len({t['temporal']['scene_token'] for t in traces}) == 1 and [t['temporal']['history_bev_present'] for t in traces] == [False,True,True,True,True],
        reset_all_recorded_fields=bool(reset) and all(all(a.values()) for a in reset),
        current_implementation_hashes_match=all(source_checks.values()),
        asset_source_hashes_match=all(asset_checks.values()),
        implementation_and_checkpoint_unchanged=report['implementation_unchanged'] and report['checkpoint_unchanged'],
        formal_admission_remains_false=acceptance['formal_evaluation_admitted'] is False)
    checks['original_evidence_unchanged'] = all(sha(RUN / rel) == digest for rel, digest in originals.items())
    summary = dict(passed=all(checks.values()), checks=checks, original_evidence_sha256=originals,
        views=views, probe_cases=cases, source_checks=source_checks, asset_checks=asset_checks,
        reset=reset, edit_journal=journal, worker_done=done,
        metrics=dict(simulation_seconds=observations[-1]['timestamp'] - observations[0]['timestamp'],
            path_length_m=float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
            initial_speed_mps=traces[0]['state_before'][3], final_speed_mps=traces[-1]['state_after'][3],
            joint_wall_seconds=report['wall_seconds'],
            minimum_all_actor_gap_m=report['minimum_all_actor_reference_gap_m'],
            minimum_managed_gap_m=min(m[0]['gap_m'] for m in managed),
            minimum_predicted_road_swept_clearance_m=min(t['predicted_road_swept_clearance_m'] for t in traces),
            hot_weather_mean_seconds=float(np.mean([r['elapsed_s'] for r in weather[1:]])),
            hot_render_weather_mean_seconds=float(np.mean([r['elapsed_total_s'] for r in weather[1:]]))),
        scope='Existing calibrated rigid SH vehicle; pre-run insert/set_pose/remove; 5 iLQR steps; rainy RGB; reset',
        limitations=['No runtime editing, external SMPL import, long-run or multi-scene certification',
                     'Proxy based on Gaussian support; no independent physical calibration',
                     'Removed input packets are verified by retained serialized-write RGB receipts and inference completion'],
        formal_evaluation_admitted=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Contact sheet of actual probe and weather outputs for visual review.
    panels = [(label, JOINT / 'managed_asset_probe' / f'{label}_CAM_FRONT.png')
              for label in ['baseline', 'inserted', 'moved', 'removed']]
    panels += [(f'{step} {kind}', JOINT / 'weather' / step / f'CAM_FRONT_{kind}.png')
               for step in ['step_000', 'step_005'] for kind in ['clear', 'weather']]
    sheet = Image.new('RGB', (960, 4 * 296), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(panels):
        with Image.open(path) as im:
            im = im.convert('RGB'); im.thumbnail((480, 270))
            x, y = i % 2 * 480, i // 2 * 296
            sheet.paste(im, (x, y + 24)); draw.text((x + 8, y + 4), label, fill='black')
    sheet.save(OUT / 'front_comparison.png')
    print(json.dumps(dict(passed=summary['passed'], checks=checks, metrics=summary['metrics']), indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
