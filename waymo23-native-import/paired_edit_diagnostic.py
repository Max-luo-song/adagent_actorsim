"""Bounded paired perception diagnostics. Never exports control/planning advice."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from edit_trust import EditPolicy, EditTrustError, _affected_roi
from evaluate_preview_vehicle import evaluate, packet_path


def asset_corners(asset):
    from asset_geometry import source_corners
    return source_corners(asset)


def image_change_audit(edited_path, baseline_path, asset_path):
    edited = json.loads(Path(edited_path).read_text())
    baseline = json.loads(Path(baseline_path).read_text())
    asset = json.loads(Path(asset_path).read_text())
    if len(edited['frames']) != len(baseline['frames']):
        raise ValueError('Different paired frame counts')
    points = asset_corners(asset)
    request = dict(kind='insert', new_bounds=[points.min(0).tolist(), points.max(0).tolist()])
    records = []
    for a, b in zip(edited['frames'], baseline['frames']):
        if a['frame'] != b['frame']:
            raise ValueError('Paired image frame mismatch')
        with packet_path(a, edited_path).open('rb') as stream:
            obs, _ = pickle.load(stream)
        with packet_path(b, baseline_path).open('rb') as stream:
            control, _ = pickle.load(stream)
        evidence = packet_path(a, edited_path).parent/'render_evidence.npz'
        with np.load(evidence, allow_pickle=False) as arrays:
            for camera, rgb in obs['rgb'].items():
                original = control['rgb'][camera]
                if rgb.shape != original.shape:
                    raise ValueError('Paired image shape mismatch')
                difference = np.max(np.abs(rgb.astype(float)-original.astype(float)), axis=2)
                row = dict(frame=a['frame'], camera=camera, changed_pixel_fraction=float((difference > 2).mean()))
                try:
                    roi = _affected_roi(request, arrays[camera+'__K'], arrays[camera+'__camera_to_world'],
                                        rgb.shape[:2], EditPolicy().halo_m)
                except EditTrustError as exc:
                    row.update(roi_usable=False, reason=str(exc))
                else:
                    row.update(roi_usable=True, projected_edit_halo_pixels=int(roi.sum()),
                               changed_fraction_inside_halo=float((difference[roi] > 2).mean()) if roi.any() else None,
                               changed_fraction_outside_halo=float((difference[~roi] > 2).mean()) if (~roi).any() else None)
                    alpha, depth = arrays[camera+'__alpha'], arrays[camera+'__depth']
                    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= EditPolicy().min_alpha) & (alpha <= 1)
                    row['combined_scene_valid_opaque_fraction_in_halo'] = float(valid[roi].mean()) if roi.any() else None
                records.append(row)
    return dict(records=records, pixel_change_threshold_uint8=2,
                combined_scene_alpha_does_not_certify_inserted_object=True,
                independent_reference_compared=False, n1_edit_certified=False)


def compare(pair_path, output):
    pair_path, output = Path(pair_path).resolve(), Path(output).resolve()
    pair = json.loads(pair_path.read_text())
    allowed = {'scene', 'edited_sequence', 'baseline_sequence', 'asset_report',
               'edited_inference', 'baseline_inference', 'expected_implementation_hash'}
    if set(pair) != allowed:
        raise ValueError('Unexpected paired diagnostic fields')
    for key in allowed-{'scene', 'expected_implementation_hash'}:
        if BASE not in Path(pair[key]).resolve().parents:
            raise ValueError('Paired diagnostic paths must stay inside project')
    sequence = json.loads(Path(pair['edited_sequence']).read_text())
    if not 1 <= len(sequence['frames']) <= 10:
        raise ValueError('Paired diagnostic frame budget exceeded')
    edited_run = json.loads((Path(pair['edited_inference'])/'run_report.json').read_text())
    baseline_run = json.loads((Path(pair['baseline_inference'])/'run_report.json').read_text())
    for run in (edited_run, baseline_run):
        if (not run.get('diagnostic_preview_only') or run.get('planning_outputs_exported') is not False
                or run.get('control_actions_executed') is not False
                or run.get('raw_scores_modified') is not False
                or run.get('implementation_unchanged_during_run') is not True
                or run['input_binding']['implementation_hash'] != pair['expected_implementation_hash']):
            raise ValueError('Run is not a frozen, control-free diagnostic')
    if (edited_run.get('diagnostic_variant', 'reference') != baseline_run.get('diagnostic_variant', 'reference')
            or edited_run['input_binding']['rules_hash'] != baseline_run['input_binding']['rules_hash']
            or edited_run['input_binding']['checkpoint'] != baseline_run['input_binding']['checkpoint']
            or edited_run['thresholds'] != baseline_run['thresholds']):
        raise ValueError('Paired inference model/rules differ')
    for folder in (pair['edited_inference'], pair['baseline_inference']):
        for index in range(len(sequence['frames'])):
            with np.load(Path(folder)/f'frame_{index:04d}/perception.npz', allow_pickle=False) as arrays:
                if any(k in arrays for k in ('trajectory', 'occupancy', 'controls', 'action')):
                    raise ValueError('Quarantined diagnostic exported forbidden planning output')
    output.mkdir(parents=True, exist_ok=False)
    edited = evaluate(pair['edited_sequence'], pair['asset_report'], pair['edited_inference'], output/'edited_matches.json')
    baseline = evaluate(pair['edited_sequence'], pair['asset_report'], pair['baseline_inference'],
                        output/'baseline_matches.json', pair['baseline_sequence'])
    records = []
    for a, b in zip(edited['records'], baseline['records']):
        row = dict(frame=a['frame'], timestamp=a['timestamp'])
        for name in ('candidates', 'retained'):
            row[name] = dict(edited_match=a[name]['geometric_match_found'],
                baseline_match=b[name]['geometric_match_found'],
                match_only_in_edited=a[name]['geometric_match_found'] and not b[name]['geometric_match_found'],
                edited_best_iou=a[name]['best_bev_iou'], baseline_best_iou=b[name]['best_bev_iou'],
                edited_matches=a[name]['matches'], baseline_matches=b[name]['matches'])
        records.append(row)
    image_audit = image_change_audit(pair['edited_sequence'], pair['baseline_sequence'], pair['asset_report'])
    (output/'paired_image_changes.json').write_text(json.dumps(image_audit, indent=2, allow_nan=False))
    report = dict(scene=pair['scene'], pair_manifest=str(pair_path), records=records,
                  complete_frames=edited['complete_frames'] and baseline['complete_frames'],
                  matching_rule=edited['matching_rule'], reference_is_placement_proxy=True,
                  paired_pose_calibration_and_packet_binding_checked=True,
                  same_inference_implementation_and_rules=True, raw_scores_modified=False,
                  control_actions_executed=False, planning_outputs_exported=False,
                  detection_accuracy_certified=False, tracking_identity_certified=False,
                  decision_correctness_certified=False, n1_edit_certified=False,
                  interpretation='Geometric car-box response at declared placement; not identity/visibility or driving correctness')
    (output/'paired_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.pair, args.output)
    print(json.dumps(result, indent=2))
