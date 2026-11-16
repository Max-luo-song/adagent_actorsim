"""CPU causal audits using frozen predictions, saved cameras and source labels.

Native associations are diagnostic, not held-out accuracy measurements: source
annotations/sparse returns do not establish full camera visibility or N1 trust.
"""
import argparse
import ast
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import BOX_CONVENTION, box_corners, box_footprint
from input_geometry import camera_projection
from evaluate_preview_vehicle import B, T, polygon_iou, packet_path, evaluate
from audit_native_vehicle_evidence import box_membership
from perception_run_binding import file_fingerprint

A = B@T


def load_old_polygon(path):
    tree = ast.parse(Path(path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'box_polygon')
    scope = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), scope)
    return scope['box_polygon']


def installed_corner_proof():
    import torch
    folder = BASE/'code/mmdetection3d-uniad-legacy/mmdet3d/core/bbox/structures'
    utils = ast.parse((folder/'utils.py').read_text())
    rotate = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == 'rotation_3d_in_axis')
    tree = ast.parse((folder/'lidar_box3d.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LiDARInstance3DBoxes')
    corners = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'corners')
    corners.decorator_list = []
    scope = dict(torch=torch, np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rotate, corners], type_ignores=[])), str(folder), 'exec'), scope)
    samples = np.array([[3., 8., -1., 1.5, 4.3, 1.7, .37], [-4., 2., .2, 2.1, 6.2, 2.5, -.83]])
    tensor = torch.as_tensor(samples, dtype=torch.float64)
    actual = scope['corners'](SimpleNamespace(tensor=tensor, dims=tensor[:, 3:6])).numpy()
    expected = np.stack([box_corners(box) for box in samples])
    return dict(box_convention=BOX_CONVENTION, actual_library_sources=[file_fingerprint(folder/name) for name in ('utils.py', 'lidar_box3d.py')],
                maximum_corner_error_m=float(np.max(np.abs(actual-expected))),
                oracle='Unmodified installed corners and rotation function bodies executed on CPU tensors')


def camera_chain(sequence_path):
    sequence = json.loads(Path(sequence_path).read_text())
    rows = []
    for frame in sequence['frames']:
        packet = packet_path(frame, sequence_path)
        with packet.open('rb') as stream:
            obs, info = pickle.load(stream)
        lidar = np.eye(4)
        lidar[:3, :3], lidar[:3, 3] = info['explicit_l2g_r_mat'], info['explicit_l2g_t']
        with np.load(packet.parent/'render_evidence.npz', allow_pickle=False) as evidence:
            for name, image in obs['rgb'].items():
                camera = info['cam_params'][name]
                c2s = evidence[name+'__camera_to_world']
                expected_l2c = np.linalg.inv(c2s)@np.linalg.inv(A)@lidar
                k, projection = camera_projection(camera, image.shape)
                resize = np.diag([1600/image.shape[1], 900/image.shape[0], 1.])
                expected_k = resize@evidence[name+'__K']
                camera_points = np.array([[0.,0.,8.,1.], [.7,-.2,10.,1.]])
                source = camera_points@c2s.T
                virtual = source@(np.linalg.inv(lidar)@A).T
                pixels = virtual@projection.T
                reference_pixels = camera_points[:, :3]@expected_k.T
                pixel_error = np.max(np.abs(pixels[:, :2]/pixels[:, 2:3]-reference_pixels[:, :2]/reference_pixels[:, 2:3]))
                rows.append(dict(frame=frame['frame'], camera=name,
                    extrinsic_matrix_max_error=float(np.max(np.abs(expected_l2c-camera['l2c']))),
                    intrinsic_matrix_max_error=float(np.max(np.abs(expected_k-k[:3, :3]))),
                    projection_pixel_max_error=float(pixel_error),
                    source_resolution_wh=[image.shape[1], image.shape[0]],
                    old_parser_focal_scale_ratio_xy=[2*image.shape[1]/1600, 2*image.shape[0]/900]))
    return dict(records=rows, internally_consistent=all(r['extrinsic_matrix_max_error'] < 1e-6 and
                    r['intrinsic_matrix_max_error'] < 1e-6 and r['projection_pixel_max_error'] < 1e-6 for r in rows),
                independent_physical_calibration_verified=False,
                interpretation='Checks internal coordinate consistency against saved renderer; not external calibration truth')


def native_matches(sequence_path, inference, polygon, index_offset=0):
    sequence = json.loads(Path(sequence_path).read_text())
    source = json.loads(Path(sequence['source_report']).read_text())
    data = Path(source.get('data_dir', source.get('original_data', source['inputs'])))
    inputs = Path(source['inputs'])
    start = int(source.get('start_timestep', 0))
    origin = np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt')
    annotations = json.loads((data/'instances/instances_info.json').read_text())
    records = []
    for index, frame in enumerate(sequence['frames']):
        with packet_path(frame, sequence_path).open('rb') as stream:
            obs, info = pickle.load(stream)
        lidar = np.eye(4)
        lidar[:3, :3], lidar[:3, 3] = info['explicit_l2g_r_mat'], info['explicit_l2g_t']
        source_to_virtual = np.linalg.inv(lidar)@A
        raw_frame = start+frame['frame']
        ego = np.linalg.inv(origin)@np.loadtxt(data/'ego_pose'/f'{raw_frame:03d}.txt')
        raw = np.fromfile(data/'lidar'/f'{raw_frame:03d}.bin', dtype=np.float32).reshape(-1,14)
        xyz = raw[:,3:6]@ego[:3,:3].T+ego[:3,3]
        valid = np.isfinite(xyz).all(1)
        xyz, ground = xyz[valid], raw[valid,10]
        owners = np.zeros(len(xyz), dtype=np.int32)
        boxes = []
        for identity, annotation in annotations.items():
            ann = annotation['frame_annotations']
            if raw_frame not in ann['frame_idx']:
                continue
            j = ann['frame_idx'].index(raw_frame)
            pose = np.linalg.inv(origin)@np.asarray(ann['obj_to_world'][j])
            size = np.asarray(ann['box_size'][j])
            mask = box_membership(xyz, pose, size)
            owners += mask
            if annotation['class_name'] == 'Vehicle':
                boxes.append((identity, pose, size, mask))
        targets = []
        for identity, pose, size, mask in boxes:
            if int((mask & (ground == 0) & (owners == 1)).sum()) < 10:
                continue
            length, width, height = size
            local = np.array([[x,y,-height/2,1.] for x,y in
                              [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]])
            transformed = local@(source_to_virtual@pose).T
            center = transformed[:,:2].mean(0)
            if np.linalg.norm(center) > 60.:
                continue
            visible = False
            for name, image in obs['rgb'].items():
                _, p = camera_projection(info['cam_params'][name], image.shape)
                pixels = transformed@p.T
                z = pixels[:,2]
                uv = pixels[:,:2]/np.maximum(z[:,None],1e-8)
                visible |= bool(((z>.1)&(uv[:,0]>=0)&(uv[:,0]<1600)&(uv[:,1]>=0)&(uv[:,1]<900)).any())
            if visible:
                targets.append((identity, transformed[:,:2]))
        with np.load(Path(inference)/f'frame_{index+index_offset:04d}/perception.npz', allow_pickle=False) as prediction:
            candidates = [(i,b,float(s),int(l)) for i,(b,s,l) in enumerate(zip(prediction['candidate_boxes'],prediction['candidate_scores'],prediction['candidate_labels']))
                          if l in (0,1,2,3,4) and s>=.4 and np.isfinite(b).all() and (b[3:6]>0).all()]
        overlaps = np.zeros((len(targets),len(candidates)))
        for i,(_,truth) in enumerate(targets):
            for j,(_,box,_,_) in enumerate(candidates):
                if np.linalg.norm(box[:2]-truth.mean(0)) < 10.:
                    overlaps[i,j] = polygon_iou(truth,polygon(box))
        assignments = []
        if overlaps.size:
            a,b = linear_sum_assignment(-overlaps)
            for i,j in zip(a,b):
                if overlaps[i,j] < .25:
                    continue
                identity,truth = targets[i]
                prediction_index,box,score,label = candidates[j]
                delta = box[:2]-truth.mean(0)
                assignments.append(dict(instance_key=identity,prediction_index=prediction_index,label=label,
                    raw_score=score,bev_iou=float(overlaps[i,j]),center_error_m=float(np.linalg.norm(delta)),
                    signed_xy_error_m=delta.tolist()))
        records.append(dict(frame=frame['frame'],frustum_and_sparse_supported_targets=len(targets),
                            high_score_vehicle_candidates=len(candidates),matches=assignments))
    residuals = np.asarray([m['signed_xy_error_m'] for r in records for m in r['matches']])
    return dict(records=records,match_count=len(residuals),
                median_signed_xy_error_m=np.median(residuals,axis=0).tolist() if len(residuals) else None,
                median_center_error_m=float(np.median(np.linalg.norm(residuals,axis=1))) if len(residuals) else None,
                matching_rule='one-to-one max IoU, vehicle labels 0..4, score>=0.4, IoU>=0.25, >=10 unique non-ground source points',
                camera_occlusion_verified=False, heldout_accuracy_certified=False,
                source_annotations=file_fingerprint(data/'instances/instances_info.json'))


def run(manifest_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    old_polygon = load_old_polygon(manifest['old_geometry_source'])
    report = dict(library_proof=installed_corner_proof(), cases=[], n1_edit_certified=False,
                  remaining_center_bias_cause_proven=False, actual_unseen_scene_generalization_proven=False)
    for case in manifest['cases']:
        camera = camera_chain(case['geometry_sequence'])
        before = native_matches(case['baseline_sequence'], case['baseline_inference'], old_polygon)
        after = native_matches(case['baseline_sequence'], case['baseline_inference'], box_footprint)
        edited = evaluate(case['historical_edited_sequence'], case['historical_asset_report'],
                          case['historical_edited_inference'], output/f'scene_{case["scene"]}_corrected_inserted_matches.json')
        report['cases'].append(dict(scene=case['scene'],camera_chain=camera,native_before=before,native_after=after,
                                     corrected_inserted_matches=str(output/f'scene_{case["scene"]}_corrected_inserted_matches.json')))
        print('Scene',case['scene'],'native matches',before['match_count'],'->',after['match_count'],
              'new median XY residual',after['median_signed_xy_error_m'],flush=True)
    (output/'causal_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
