"""Prepare scene-specific evaluation assets; deliberately never dispatch controls."""

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
import fcntl
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import subprocess
import sys
import tempfile
import time
import traceback

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[1]
DEFAULT_OUT = ROOT / 'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/patch_generalization_assets_v1'
CASES = _hugsim_expand(json.loads((LOCAL / 'cases.json').read_text()))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify():
    contract = _hugsim_expand(json.loads((LOCAL / 'contract.json').read_text()))
    groups = [contract['donor_pins'], _hugsim_expand(json.loads((LOCAL / 'source_pins.json').read_text()))]
    groups += [arm['code_pins'] for arm in contract['arms'].values()]
    for group in groups:
        for path, expected in group.items():
            p = Path(path)
            if not p.is_absolute():
                p = LOCAL / p
            if sha(p) != expected:
                raise ValueError('Frozen source changed: ' + str(p))
    return contract


def audit():
    import numpy as np
    result = {'schema': 'cross_scene_source_audit_v1', 'cases': {}, 'new_closed_loop_executed': False,
              'global_patch_validated': False, 'gpu_executed': False}
    source = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph'))
    listing = (source / 'Omniscenegraph/data/waymo_train_list.txt').read_text().splitlines()
    weights = source / 'Omniscenegraph/third_party/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth'
    result['semantic_weights_available'] = weights.is_file()
    for key, case in CASES.items():
        raw = Path(case['raw'])
        expected = list(range(case['frames']))
        ids = sorted(int(p.stem) for p in (raw / 'ego_pose').glob('*.txt'))
        errors = []
        if ids != expected:
            errors.append('pose frame coverage mismatch')
        for frame in expected:
            for folder, extension, suffixes in [('ego_pose', 'txt', ['']), ('lidar', 'bin', ['']),
                                               ('images', 'jpg', ['_' + str(i) for i in range(5)]),
                                               ('road_masks', 'png', ['_' + str(i) for i in range(5)]),
                                               ('sky_masks', 'png', ['_' + str(i) for i in range(5)])]:
                for suffix in suffixes:
                    path = raw / folder / ('%03d%s.%s' % (frame, suffix, extension))
                    if not path.is_file():
                        errors.append('missing ' + str(path))
        for camera in range(5):
            for folder, shape in [('intrinsics', (9,)), ('extrinsics', (4, 4))]:
                path = raw / folder / ('%d.txt' % camera)
                if not path.is_file() or np.loadtxt(path).shape != shape:
                    errors.append('invalid calibration ' + str(path))
        tfrecord = raw.parents[2] / 'raw' / (listing[case['scene']] + '.tfrecord')
        checkpoint = Path(case['checkpoint'])
        for path in [checkpoint, tfrecord, weights]:
            if not path.is_file():
                errors.append('missing ' + str(path))
        result['cases'][key] = dict(scene=case['scene'], frames=len(ids), source_files_present=not errors,
                                   errors=errors, raw_record=str(tfrecord), checkpoint=str(checkpoint),
                                   identity_verified=False, native_evaluation_admitted=False,
                                   raw_processed_identity_check='pending metadata stage',
                                   remaining=['full source identity', 'full semantic export', 'point transfer',
                                              'reviewed map and runtime/scoring adapter', 'fresh closed loops'])
    return result


def check_bundle(bundle, scene):
    meta = _hugsim_expand(json.loads((bundle / 'source_manifest.json').read_text()))
    case = CASES[str(scene)]
    if meta['scene'] != scene or meta['frames'] != case['frames'] or meta['raw_data'] != case['raw'] or meta['source_checkpoint'] != case['checkpoint']:
        raise ValueError('Bundle scene identity mismatch')
    if sha(case['checkpoint']) != meta['source_checkpoint_sha256']:
        raise ValueError('Changed source checkpoint')
    for filename, expected in meta['artifacts'].items():
        if sha(bundle / filename) != expected:
            raise ValueError('Changed source bundle ' + filename)
    return meta


def run_case(args):
    scene = args.scene
    os.environ['GENERALIZATION_SCENE'] = str(scene)
    import scene_data
    case_root = args.output / ('scene%d' % scene)
    case_root.mkdir(parents=True, exist_ok=True)
    lock = (case_root / 'assets.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundle = case_root / 'bundle'
    stages = ['prepare', 'metadata', 'semantics', 'points'] if args.stage == 'assets' else [args.stage]
    try:
        for stage in stages:
            atomic(case_root / 'status.json', dict(stage=stage, scene=scene, pid=os.getpid(),
                   updated_epoch=time.time(), new_closed_loop_executed=False, native_evaluation_admitted=False))
            if stage == 'prepare':
                if not bundle.exists():
                    # Stage directory is unique and retained on error for diagnosis.
                    temporary = Path(tempfile.mkdtemp(prefix='prepare_', dir=case_root)) / 'bundle'
                    scene_data.prepare(temporary)
                    temporary.rename(bundle)
                    temporary.parent.rmdir()
                check_bundle(bundle, scene)
                continue
            check_bundle(bundle, scene)
            if stage == 'metadata':
                target = bundle / 'raw_metadata'
                if not target.exists():
                    from raw_metadata import export
                    temporary = Path(tempfile.mkdtemp(prefix='metadata_', dir=case_root))
                    (temporary / 'source_manifest.json').symlink_to(bundle / 'source_manifest.json')
                    export(temporary)
                    (temporary / 'raw_metadata').rename(target)
                    (temporary / 'source_manifest.json').unlink()
                    temporary.rmdir()
                meta = _hugsim_expand(json.loads((target / 'raw_metadata.json').read_text()))
                if meta['scene_index'] != scene or len(meta['frames']) != CASES[str(scene)]['frames']:
                    raise ValueError('Raw identity coverage mismatch')
                if sha(meta['raw_record']) != meta['raw_record_sha256']:
                    raise ValueError('Changed raw record')
            elif stage == 'semantics':
                from export_semantics import export
                # Export validates every existing map and supports exact-contract resume.
                export(bundle, args.device)
            elif stage == 'points':
                target = bundle / 'point_semantics'
                if not target.exists():
                    from point_semantics import transfer
                    temporary = Path(tempfile.mkdtemp(prefix='points_', dir=case_root))
                    for name in ['source_manifest.json', 'semantic19', 'raw_metadata', 'poses_and_clock.npz', 'background_sr_world.npy']:
                        (temporary / name).symlink_to(bundle / name)
                    transfer(temporary)
                    (temporary / 'point_semantics').rename(target)
                    for path in temporary.iterdir():
                        path.unlink()
                    temporary.rmdir()
                point = _hugsim_expand(json.loads((target / 'point_semantics_manifest.json').read_text()))
                if len(point['frames']) != CASES[str(scene)]['frames'] or point['unknown_is_free']:
                    raise ValueError('Invalid point semantics coverage')
        atomic(case_root / 'status.json', dict(stage='requested_assets_completed_review_pending', scene=scene,
               completed_stages=stages, updated_epoch=time.time(), new_closed_loop_executed=False,
               native_evaluation_admitted=False, global_patch_validated=False,
               missing=['reviewed scene-specific map', 'runtime/scoring adapter', 'fresh paired closed loops']))
    except BaseException as exc:
        atomic(case_root / 'status.json', dict(stage='error', scene=scene, error=repr(exc),
               traceback=traceback.format_exc(), new_closed_loop_executed=False, native_evaluation_admitted=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'prepare', 'metadata', 'semantics', 'points', 'assets', 'status'])
    parser.add_argument('--scene', type=int, choices=[int(s) for s in CASES])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    verify()
    if args.stage == 'status':
        results = {p.parent.name: _hugsim_expand(json.loads(p.read_text())) for p in args.output.glob('scene*/status.json')}
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.stage == 'audit':
        report = audit()
        atomic(args.output / 'source_audit.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not all(x['source_files_present'] for x in report['cases'].values()):
            raise SystemExit(1)
    elif args.scene is None:
        for scene in CASES:
            subprocess.run([sys.executable, '-B', '-u', str(LOCAL / 'pipeline.py'), args.stage,
                            '--scene', scene, '--output', str(args.output), '--device', args.device], check=True)
    else:
        run_case(args)


if __name__ == '__main__':
    main()
