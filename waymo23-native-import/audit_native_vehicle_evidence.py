"""Rank sensor-backed native vehicles without promoting sparse data to N1 truth.

All thresholds are shared diagnostic selection rules, not edit admission rules.
No detector scores, Gaussian renders or generated geometry are used here.
"""
import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from collect_n1_sensor_evidence import (fingerprint, project, CV_TO_WAYMO,
                                       projection_on_optical_branch)


@dataclass(frozen=True)
class CandidateAuditPolicy:
    depth_cell_pixels: int = 8
    occlusion_absolute_m: float = .15
    occlusion_relative: float = .02
    minimum_points_per_view: int = 10
    minimum_cells_per_view: int = 6
    minimum_same_frame_cameras: int = 2
    minimum_supported_frames: int = 3


def configured_holdout_neighbors(anchor, count, stride):
    if not isinstance(stride, int) or stride < 0:
        raise ValueError('Invalid configured test stride')
    if stride == 0:
        return []
    heldout = np.arange(stride, count, stride, dtype=int)
    # Include the exact frame, if present, plus nearest strict neighbor on each side.
    return sorted(set(([int(heldout[heldout < anchor][-1])] if (heldout < anchor).any() else [])
                      + ([anchor] if anchor in heldout else [])
                      + ([int(heldout[heldout > anchor][0])] if (heldout > anchor).any() else [])))


def split_configuration(audit_path):
    audit = json.loads(Path(audit_path).read_text())
    source = json.loads(Path(audit['source_report']).read_text())
    inputs = Path(source['inputs'])
    cfg_path = inputs/'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    stride = cfg['data']['pixel_source']['test_image_stride']
    code_path = inputs/'backup/datasets/driving_dataset.py'
    tree = ast.parse(code_path.read_text())
    funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name == 'split_train_test']
    expected = 'np.arange(self.data_cfg.pixel_source.test_image_stride, self.num_img_timesteps, self.data_cfg.pixel_source.test_image_stride)'
    recognized = len(funcs) == 1 and any(isinstance(n, ast.Call) and ast.unparse(n) == expected
                                       for n in ast.walk(funcs[0]))
    if not recognized:
        raise ValueError('Archived split implementation is not the supported stride rule')
    return dict(test_image_stride=stride, config=fingerprint(cfg_path),
                archived_split_code=fingerprint(code_path),
                configured_holdout_frames_near_anchor=configured_holdout_neighbors(
                    audit['automatic_start_frame'], int(source['frames']), stride),
                actual_training_exclusion_independently_verified=False,
                lidar_training_exclusion_verified=False,
                interpretation='configuration-designated image holdout only; not independent geometry')


def box_membership(points, pose, size):
    pose, size = np.asarray(pose, dtype=float), np.asarray(size, dtype=float)
    if (pose.shape != (4, 4) or size.shape != (3,) or not np.isfinite(pose).all()
            or not np.isfinite(size).all() or np.any(size <= 0)
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Invalid annotated object pose/size')
    return (np.abs((points-pose[:3, 3])@pose[:3, :3]) <= size/2).all(axis=1)


def sparse_view_support(uv, depth, target_mask, eligible_mask, image_wh, policy):
    """Nearest competing LiDAR depth in coarse cells detects contradictions.

    No closer return means UNKNOWN visibility, never proof of a clear camera ray.
    """
    width, height = map(int, image_wh)
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > .1)
    inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    cell = policy.depth_cell_pixels
    grid_w, grid_h = (width+cell-1)//cell, (height+cell-1)//cell
    indices = np.flatnonzero(inside)
    keys = np.full(len(depth), -1, dtype=int)
    pixels = np.floor(uv[indices]/cell).astype(int)
    keys[indices] = pixels[:, 1]*grid_w+pixels[:, 0]
    competing = inside & ~target_mask
    nearest = np.full(grid_w*grid_h, np.inf)
    np.minimum.at(nearest, keys[competing], depth[competing])
    target = np.flatnonzero(inside & eligible_mask)
    tolerance = policy.occlusion_absolute_m+policy.occlusion_relative*depth[target]
    contradicted = nearest[keys[target]] < depth[target]-tolerance
    supported = target[~contradicted]
    cells = int(len(np.unique(keys[supported])))
    return dict(projected_eligible_points=len(target), contradicted_points=int(contradicted.sum()),
                uncontradicted_points=len(supported), uncontradicted_cells=cells,
                selected_for_diagnostic=(len(supported) >= policy.minimum_points_per_view
                                         and cells >= policy.minimum_cells_per_view),
                visibility_proven=False, dense_foreground_mask_available=False), supported


def review(manifest_path, output, policy=None, export_count=3):
    policy = policy or CandidateAuditPolicy()
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    verified = {}

    def verify(record):
        path = record['path']
        if path not in verified:
            actual = fingerprint(path)
            if any(actual[key] != record[key] for key in ('bytes', 'sha256')):
                raise ValueError('Source fingerprint mismatch: '+path)
            verified[path] = actual

    verify(manifest['sensor_audit'])
    verify(manifest['annotation_source'])
    audit = json.loads(Path(manifest['sensor_audit']['path']).read_text())
    source = json.loads(Path(audit['source_report']).read_text())
    start = int(source.get('start_timestep', 0))
    origin_path = Path(source['inputs'])/'ego_pose'/f'{start:03d}.txt'
    origin = np.loadtxt(origin_path)
    annotations = json.loads(Path(manifest['annotation_source']['path']).read_text())
    split = split_configuration(manifest['sensor_audit']['path'])
    stride = split['test_image_stride']
    candidates = {item['instance_key']: dict(instance_key=item['instance_key'],
                   source_id=item['source_id'], views=[], frames=[])
                  for item in manifest['native_vehicle_candidates']}
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    frame_records = []
    crop_jobs = []
    for frame in sorted({v['frame'] for v in manifest['views']}):
        records = [v for v in manifest['views'] if v['frame'] == frame]
        for record in records:
            for key in ('image', 'lidar', 'ego_pose', 'intrinsic', 'extrinsic'):
                verify(record[key])
        record = records[0]
        raw = np.fromfile(record['lidar']['path'], dtype=np.float32)
        if raw.size % 14:
            raise ValueError('Invalid raw LiDAR schema')
        raw = raw.reshape(-1, 14)
        pose = np.linalg.inv(origin)@np.loadtxt(record['ego_pose']['path'])
        xyz = raw[:, 3:6]@pose[:3, :3].T+pose[:3, 3]
        valid = np.isfinite(xyz).all(axis=1)
        xyz, ground = xyz[valid], raw[valid, 10]
        with np.load(manifest_path.parent/f'frame_{frame:03d}/lidar_source.npz', allow_pickle=False) as saved:
            if (saved['xyz'].shape != xyz.shape or not np.allclose(saved['xyz'], xyz, atol=1e-7)
                    or not np.array_equal(saved['ground_label'], ground)):
                raise ValueError('Derived sensor cloud does not match raw source')
        masks, boxes = {}, {}
        owners = np.zeros(len(xyz), dtype=np.int32)
        for identity, annotation in annotations.items():
            ann = annotation['frame_annotations']
            if start+frame not in ann['frame_idx']:
                continue
            index = ann['frame_idx'].index(start+frame)
            obj = np.linalg.inv(origin)@np.asarray(ann['obj_to_world'][index])
            size = np.asarray(ann['box_size'][index])
            mask = box_membership(xyz, obj, size)
            owners += mask
            if identity in candidates:
                masks[identity], boxes[identity] = mask, (obj, size)
        for identity in masks:
            item = next(c for c in manifest['native_vehicle_candidates'] if c['instance_key'] == identity)
            observation = next(o for o in item['observations'] if o['raw_frame'] == start+frame)
            if (item['source_id'] != annotations[identity]['id']
                    or not np.allclose(observation['object_to_source'], boxes[identity][0], atol=1e-7)
                    or not np.allclose(observation['box_lwh'], boxes[identity][1], atol=1e-7)):
                raise ValueError('Candidate annotation binding mismatch')
        for identity, mask in masks.items():
            eligible = mask & (ground == 0) & (owners == 1)
            obj, size = boxes[identity]
            candidates[identity]['frames'].append(dict(frame=frame,
                configured_image_holdout=bool(stride and frame >= stride and frame % stride == 0),
                box_points=int(mask.sum()), ground_labeled_box_points=int((mask & (ground == 1)).sum()),
                ambiguous_box_points=int((mask & (owners > 1)).sum()),
                eligible_non_ground_points=int(eligible.sum()),
                object_center=obj[:3, 3].tolist(), box_lwh=size.tolist()))
        for record in records:
            camera = np.asarray(record['camera_to_source'])
            expected = pose@np.loadtxt(record['extrinsic']['path'])@CV_TO_WAYMO
            if not np.allclose(camera, expected, atol=1e-7):
                raise ValueError('Camera pose does not match source calibration')
            intrinsic = np.loadtxt(record['intrinsic']['path']).reshape(-1)
            with Image.open(record['image']['path']) as image:
                image_wh = image.size
            uv, depth = project(xyz, camera, intrinsic)
            in_image = ((depth > .1) & np.isfinite(uv).all(axis=1)
                        & (uv[:, 0] >= 0) & (uv[:, 0] < image_wh[0])
                        & (uv[:, 1] >= 0) & (uv[:, 1] < image_wh[1]))
            indices = np.flatnonzero(in_image)
            optical = np.zeros(len(xyz), dtype=bool)
            optical[indices] = projection_on_optical_branch(xyz[indices], camera, intrinsic,
                                                            uv[indices], depth[indices])
            # Invalid folded projections must not act as target OR occluding points.
            checked_uv = uv.copy()
            checked_uv[~optical] = np.nan
            for identity, mask in masks.items():
                eligible = mask & (ground == 0) & (owners == 1)
                metrics, supported = sparse_view_support(checked_uv, depth, mask, eligible, image_wh, policy)
                metrics['folded_in_image_target_points_rejected'] = int((eligible & in_image & ~optical).sum())
                obj, size = boxes[identity]
                corners = (signs*size/2)@obj[:3, :3].T+obj[:3, 3]
                corner_uv, corner_z = project(corners, camera, intrinsic)
                crop = None
                if projection_on_optical_branch(corners, camera, intrinsic, corner_uv, corner_z).all():
                    low = np.maximum(np.floor(corner_uv.min(0)), 0).astype(int)
                    high = np.minimum(np.ceil(corner_uv.max(0)), image_wh).astype(int)
                    if (high-low >= 16).all():
                        crop = [*low.tolist(), *high.tolist()]
                metrics.update(frame=frame, camera=record['camera'], crop_xyxy=crop,
                               configured_image_holdout=bool(stride and frame >= stride and frame % stride == 0))
                metrics['selected_for_diagnostic'] &= crop is not None
                candidates[identity]['views'].append(metrics)
                if metrics['selected_for_diagnostic']:
                    crop_jobs.append((identity, record, metrics, uv[supported].copy(), xyz[supported].copy()))
        frame_records.append(dict(frame=frame, raw_frame=start+frame, raw_points=len(raw),
                                  finite_points=len(xyz), overlapping_annotation_points=int((owners > 1).sum()),
                                  raw_to_derived_cloud_verified=True, camera_poses_verified=True))
        print('Audited scene', manifest['scene'], 'frame', frame, flush=True)
    ranked = []
    for candidate in candidates.values():
        views = [v for v in candidate['views'] if v['selected_for_diagnostic']]
        frame_cameras = {f['frame']: {v['camera'] for v in views if v['frame'] == f['frame']}
                         for f in candidate['frames']}
        multi = [f for f, cameras in frame_cameras.items() if len(cameras) >= policy.minimum_same_frame_cameras]
        heldout_multi = [f for f in multi if stride and f >= stride and f % stride == 0]
        candidate.update(supported_same_frame_multicamera_frames=multi,
                         configured_holdout_multicamera_frames=heldout_multi,
                         distinct_supported_cameras=len({v['camera'] for v in views}),
                         diagnostic_multiview_candidate=len(multi) >= policy.minimum_supported_frames,
                         configured_holdout_multiview_candidate=bool(heldout_multi),
                         real_wheel_contacts_verified=False, foreground_segmentation_verified=False,
                         n1_edit_certified=False)
        ranked.append(candidate)
    ranked.sort(key=lambda c: (-int(c['diagnostic_multiview_candidate']),
                              -len(c['configured_holdout_multicamera_frames']),
                              -len(c['supported_same_frame_multicamera_frames']),
                              -sum(v['uncontradicted_cells'] for v in c['views'] if v['selected_for_diagnostic']),
                              c['instance_key']))
    selected = {c['instance_key'] for c in ranked[:export_count] if c['diagnostic_multiview_candidate']}
    exported = []
    for identity, record, metrics, uv, points in crop_jobs:
        if identity not in selected:
            continue
        folder = output/f'candidate_{identity}'
        folder.mkdir(exist_ok=True)
        stem = f'frame_{metrics["frame"]:03d}_camera_{record["camera"]}'
        with Image.open(record['image']['path']) as image:
            crop = image.crop(metrics['crop_xyxy']).convert('RGB')
        draw = ImageDraw.Draw(crop)
        low = np.asarray(metrics['crop_xyxy'][:2])
        for x, y in uv-low:
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 255, 0))
        crop.save(folder/(stem+'_sparse_overlay.jpg'))
        np.savez_compressed(folder/(stem+'_sensor_points.npz'), xyz_source=points, uv_raw=uv,
                            crop_xyxy=metrics['crop_xyxy'])
        exported.append(dict(instance_key=identity, frame=metrics['frame'], camera=record['camera'],
                             overlay=str(folder/(stem+'_sparse_overlay.jpg')),
                             raw_image=record['image'], sparse_points=str(folder/(stem+'_sensor_points.npz'))))
    report = dict(scene=manifest['scene'], policy=asdict(policy), source_manifest=fingerprint(manifest_path),
                  source_report=fingerprint(audit['source_report']), origin_pose=fingerprint(origin_path),
                  verified_source_files=list(verified.values()), split_audit=split, frame_audits=frame_records,
                  candidate_count=len(ranked), diagnostic_multiview_candidates=sum(c['diagnostic_multiview_candidate'] for c in ranked),
                  configured_holdout_multiview_candidates=sum(c['diagnostic_multiview_candidate'] and c['configured_holdout_multiview_candidate'] for c in ranked),
                  ranked_candidates=ranked, exported_sensor_overlays=exported,
                  n1_edit_certified=False, dense_depth_available=False, trust_domain_constructed=False,
                  sparse_points_shared_across_camera_projections=True,
                  physical_camera_visibility_verified=False,
                  blockers=['sparse points and annotated boxes are not dense foreground or complete object skin',
                            'camera/LiDAR physical synchronization and dynamic motion compensation not verified',
                            'training exclusion not independently audited; LiDAR may initialize the fitted model',
                            'real wheel contact patches and independent complete geometry unavailable',
                            'trusted baseline field and edit pre/post revalidation not yet available'],
                  interpretation='Ranking for diagnostic follow-up only; absent sparse occlusion is unknown visibility')
    (output/'native_evidence_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ('scene', 'candidate_count', 'diagnostic_multiview_candidates',
                                            'configured_holdout_multiview_candidates', 'n1_edit_certified')}, indent=2))
