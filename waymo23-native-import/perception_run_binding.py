"""Reproducible input/code binding for paired diagnostic inference."""
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def file_fingerprint(path):
    path = Path(path).resolve()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=value.hexdigest())


def implementation_binding():
    from edit_trust import digest
    files = ['waymo23-native-import/'+name for name in (
        'run_reliable_sequence.py', 'infer_three_pad32.py', 'five_uniad.py',
        'runtime_configuration.py', 'perception_preview.py', 'perception_run_binding.py',
        'asset_calibration.py', 'asset_geometry.py', 'shared_rules.py', 'n1_cross_scene_audit.py', 'diagnostic_variants.py', 'model_profiles.py')]
    files += ['code/UniAD_SIM/'+name for name in (
        'tools/closeloop/dataparser.py', 'tools/closeloop/e2e.py',
        'projects/configs/stage2_e2e/base_e2e.py',
        'projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py',
        'projects/mmdet3d_plugin/uniad/detectors/uniad_track.py',
        'projects/mmdet3d_plugin/uniad/scene_reliability.py',
        'projects/mmdet3d_plugin/uniad/box_geometry.py',
        'projects/mmdet3d_plugin/uniad/input_geometry.py',
        'projects/mmdet3d_plugin/core/bbox/util.py',
        'projects/mmdet3d_plugin/core/bbox/coders/detr3d_track_coder.py',
        'projects/mmdet3d_plugin/uniad/edit_trust.py')]
    files += ['code/mmdetection3d-uniad-legacy/mmdet3d/core/bbox/structures/'+name
              for name in ('lidar_box3d.py', 'utils.py')]
    return digest({name: file_fingerprint(BASE/name)['sha256'] for name in files})


def input_binding(sequence_path, records):
    from run_reliable_sequence import packet_path
    from shared_rules import shared_rules
    from edit_trust import digest
    import json
    from model_profiles import resolve_model_profile
    profile_name, profile = resolve_model_profile(json.loads(Path(sequence_path).read_text()))
    rules = shared_rules()
    rules['model_input'] = profile
    checkpoint = (BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth').resolve()
    stat = checkpoint.stat()
    return dict(sequence=file_fingerprint(sequence_path),
                packets=[dict(frame=r['frame'], timestamp=r['timestamp'],
                              **file_fingerprint(packet_path(r, Path(sequence_path)))) for r in records],
                rules=rules, rules_hash=digest(rules), implementation_hash=implementation_binding(),
                checkpoint=dict(path=str(checkpoint), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                full_content_hash_checked=False))
