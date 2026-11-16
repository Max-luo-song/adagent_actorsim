"""Post-hoc box matching to the declared inserted asset; never drives a model."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import BOX_CONVENTION, box_footprint

B = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
T = np.array([[0., -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])


def signed_area(poly):
    poly = np.asarray(poly)
    return float(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1))/2) if len(poly) >= 3 else 0.


def polygon_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if signed_area(a) < 0:
        a = a[::-1]
    if signed_area(b) < 0:
        b = b[::-1]
    output = list(a)
    for c, d in zip(b, np.roll(b, -1, axis=0)):
        incoming, output = output, []
        if not incoming:
            break
        edge = d-c
        cross = lambda p: edge[0]*(p[1]-c[1])-edge[1]*(p[0]-c[0])
        previous = incoming[-1]
        for point in incoming:
            p, q = cross(previous), cross(point)
            if (p >= -1e-10) != (q >= -1e-10):
                output.append(previous+(point-previous)*(p/(p-q)))
            if q >= -1e-10:
                output.append(point)
            previous = point
    intersection = abs(signed_area(output))
    return intersection/max(abs(signed_area(a))+abs(signed_area(b))-intersection, 1e-12)


def box_polygon(box):
    return box_footprint(box)


def reference_polygon(asset, info):
    pose = np.asarray(asset['placements'][0]['asset_to_source'])
    if 'local_bottom_y' in asset:
        bottom = asset['local_bottom_y']
    else:
        audit = json.loads(Path(asset['sensor_audit']).read_text())
        bottom = pose[2, 3]-audit['proposed_anchor_source_xyz'][2]
    from asset_geometry import bottom_polygon, validate_pose
    pose = validate_pose(pose)
    points = bottom_polygon(asset['asset_wlh'], bottom)
    local = np.column_stack([points, np.ones(4)])
    lidar = np.eye(4)
    lidar[:3, :3] = info['explicit_l2g_r_mat']
    lidar[:3, 3] = info['explicit_l2g_t']
    return (local@(np.linalg.inv(lidar)@B@T@pose).T)[:, :2]


def packet_path(frame, sequence_path):
    path = Path(frame['packet'])
    return path.resolve() if path.is_absolute() else (Path(sequence_path).resolve().parent/path).resolve()


def verify_prediction_binding(sequence_path, prediction_sequence_path, inference):
    """Prevent stale placement predictions or mismatched baselines being reused."""
    sequence = json.loads(Path(sequence_path).read_text())
    prediction_sequence = json.loads(Path(prediction_sequence_path).read_text())
    run = json.loads((Path(inference)/'run_report.json').read_text())
    recorded_path = Path(run['source'])
    recorded = json.loads(recorded_path.read_text())
    binding = run.get('input_binding')
    if binding:
        from perception_run_binding import file_fingerprint
        if (file_fingerprint(recorded_path)['sha256'] != binding['sequence']['sha256']
                or run.get('implementation_unchanged_during_run') is not True):
            raise ValueError('Inference source/code changed during or after inference')
        if len(binding['packets']) != len(prediction_sequence['frames']):
            raise ValueError('Incomplete input fingerprint list')
        for expected, frame in zip(binding['packets'], prediction_sequence['frames']):
            actual = file_fingerprint(packet_path(frame, prediction_sequence_path))
            if actual != {key: expected[key] for key in ('path', 'bytes', 'sha256')}:
                raise ValueError('Inference packet content changed')
    if (run.get('completed_frames') != len(prediction_sequence['frames'])
            or len(sequence['frames']) != len(prediction_sequence['frames'])
            or len(recorded['frames']) != len(prediction_sequence['frames'])):
        raise ValueError('Incomplete or differently sized prediction sequence')
    for reference, predicted, original in zip(sequence['frames'], prediction_sequence['frames'], recorded['frames']):
        if (reference['frame'] != predicted['frame'] or predicted['frame'] != original['frame']
                or not np.isclose(reference['timestamp'], predicted['timestamp'], atol=1e-8, rtol=0)
                or packet_path(predicted, prediction_sequence_path) != packet_path(original, recorded_path)):
            raise ValueError('Prediction frame/packet binding mismatch')
        with packet_path(reference, sequence_path).open('rb') as stream:
            ref_obs, ref_info = pickle.load(stream)
        with packet_path(predicted, prediction_sequence_path).open('rb') as stream:
            pred_obs, pred_info = pickle.load(stream)
        for key in ('explicit_l2g_r_mat', 'explicit_l2g_t'):
            if not np.allclose(ref_info[key], pred_info[key], atol=1e-8, rtol=0):
                raise ValueError('Paired ego pose differs')
        if (set(ref_obs['rgb']) != set(pred_obs['rgb'])
                or not np.isclose(ref_info['timestamp'], pred_info['timestamp'], atol=1e-8, rtol=0)
                or ref_info['command'] != pred_info['command']
                or not np.isclose(ref_info['ego_velo'], pred_info['ego_velo'], atol=1e-8, rtol=0)
                or not np.isclose(ref_info['ego_steer'], pred_info['ego_steer'], atol=1e-8, rtol=0)):
            raise ValueError('Paired sensor rig/clock/ego state differs')
        for name in ref_obs['rgb']:
            a, b = ref_info['cam_params'][name], pred_info['cam_params'][name]
            if (a['intrinsic'] != b['intrinsic'] or not np.allclose(a['l2c'], b['l2c'], atol=1e-8, rtol=0)
                    or ref_obs['rgb'][name].shape != pred_obs['rgb'][name].shape):
                raise ValueError('Paired camera calibration differs')
    return run


def evaluate(sequence_path, asset_path, inference, output, prediction_sequence_path=None):
    prediction_sequence_path = prediction_sequence_path or sequence_path
    run = verify_prediction_binding(sequence_path, prediction_sequence_path, inference)
    sequence = json.loads(Path(sequence_path).read_text())
    asset = json.loads(Path(asset_path).read_text())
    records = []
    for index, frame in enumerate(sequence['frames']):
        predictions = Path(inference)/f'frame_{index:04d}/perception.npz'
        if not predictions.exists():
            raise ValueError('Missing prediction frame: '+str(predictions))
        with packet_path(frame, sequence_path).open('rb') as stream:
            obs, info = pickle.load(stream)
        truth = reference_polygon(asset, info)
        row = dict(frame=frame['frame'], timestamp=frame['timestamp'], reference_polygon=truth.tolist())
        with np.load(predictions, allow_pickle=False) as data:
            if 'box_convention' in data and str(data['box_convention']) != BOX_CONVENTION:
                raise ValueError('Unsupported predicted box convention')
            for name, prefix in [('candidates', 'candidate_'), ('retained', '')]:
                matches = []
                best_iou = 0.
                valid_car_boxes = 0
                for i, (box, score, label) in enumerate(zip(data[prefix+'boxes'], data[prefix+'scores'], data[prefix+'labels'])):
                    if (label != 0 or not np.isfinite(box).all() or not np.isfinite(score)
                            or (box[3:6] <= 0).any()):
                        continue
                    valid_car_boxes += 1
                    iou = polygon_iou(truth, box_polygon(box))
                    best_iou = max(best_iou, iou)
                    if iou >= .25:
                        predicted = box_polygon(box)
                        reference_axis, predicted_axis = truth[1]-truth[0], predicted[2]-predicted[1]
                        cosine = abs(reference_axis@predicted_axis)/(np.linalg.norm(reference_axis)*np.linalg.norm(predicted_axis))
                        matches.append(dict(index=i, raw_score=float(score), bev_iou=iou,
                            center_error_m=float(np.linalg.norm(predicted.mean(0)-truth.mean(0))),
                            heading_error_modulo_180_degrees=float(np.degrees(np.arccos(np.clip(cosine, 0, 1)))),
                            box=box.tolist()))
                matches.sort(key=lambda v: v['raw_score'], reverse=True)
                row[name] = dict(geometric_match_found=bool(matches), best_bev_iou=best_iou,
                                 valid_car_boxes=valid_car_boxes, matches=matches[:5])
        records.append(row)
    report = dict(sequence_report=str(sequence_path), asset_report=str(asset_path), inference=str(inference),
                  matching_rule='vehicle class 0 and oriented BEV IoU >= 0.25; shared across scenes', records=records,
                  oracle_used_only_after_inference=True, ground_truth_supplied_to_model=False,
                  raw_scores_modified=False, confidence_calibrated=False,
                  prediction_sequence_report=str(prediction_sequence_path), prediction_binding_checked=True,
                  complete_frames=len(records)==len(sequence['frames']),
                  prediction_diagnostic_only=run.get('diagnostic_preview_only'),
                  reference_is_declared_placement_proxy=True,
                  box_convention=BOX_CONVENTION,
                  visibility_and_occlusion_validated=False, detection_accuracy_validated=False,
                  closed_loop_run=False, n1_edit_certified=False, generalization_proven=False)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-report', required=True)
    parser.add_argument('--asset-report', required=True)
    parser.add_argument('--inference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prediction-sequence-report')
    args = parser.parse_args()
    evaluate(args.sequence_report, args.asset_report, args.inference, args.output, args.prediction_sequence_report)
