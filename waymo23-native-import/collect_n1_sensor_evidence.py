"""Collect bounded real sensor references; do not issue an edit certificate.

Raw-camera references and sparse LiDAR are kept separate from synthetic edited
renders. Native annotated vehicles provide candidate sensor-backed assets, not
an identity match to the independently downloaded 3DRealCar library.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CV_TO_WAYMO = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.]])


def fingerprint(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=value.hexdigest())


def project(points, camera_to_source, intrinsic):
    """Waymo processed raw images: Brown-Conrady k1,k2,p1,p2,k3."""
    local = (points-camera_to_source[:3, 3])@camera_to_source[:3, :3]
    z = local[:, 2]
    xy = local[:, :2]/np.maximum(z[:, None], 1e-8)
    x, y = xy.T
    k1, k2, p1, p2, k3 = intrinsic[4:9]
    r2 = x*x+y*y
    with np.errstate(over='ignore', invalid='ignore'):
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        distorted = np.column_stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                                     y*radial+p1*(r2+2*y*y)+2*p2*x*y])
    return distorted*intrinsic[:2]+intrinsic[2:4], z


def projection_on_optical_branch(points, camera_to_source, intrinsic, uv=None, depth=None):
    """Reject Brown-Conrady folds by inversion from the central optical branch.

    A polynomial can project an extreme off-axis ray back inside the image.
    Pixel bounds alone therefore do not establish physical camera visibility.
    This diagnostic checks positive local Jacobians and inverse consistency;
    it does not certify lens calibration, occlusion or sensor synchronization.
    """
    if uv is None or depth is None:
        uv, depth = project(points, camera_to_source, intrinsic)
    local = (points-camera_to_source[:3, 3])@camera_to_source[:3, :3]
    normalized = local[:, :2]/np.maximum(local[:, 2:3], 1e-8)
    target = (uv-intrinsic[2:4])/intrinsic[:2]
    estimate = target.copy()
    valid = (depth > .1) & np.isfinite(target).all(axis=1) & np.isfinite(normalized).all(axis=1)
    k1, k2, p1, p2, k3 = intrinsic[4:9]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for _ in range(20):
            x, y = estimate.T
            r2 = x*x+y*y
            radial = 1+k1*r2+k2*r2**2+k3*r2**3
            derivative = k1+2*k2*r2+3*k3*r2**2
            predicted = np.column_stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                                         y*radial+p1*(r2+2*y*y)+2*p2*x*y])
            jxx = radial+2*x*x*derivative+2*p1*y+6*p2*x
            jyy = radial+2*y*y*derivative+6*p1*y+2*p2*x
            jxy = 2*x*y*derivative+2*p1*x+2*p2*y
            determinant = jxx*jyy-jxy*jxy
            valid &= np.isfinite(determinant) & (jxx > 1e-8) & (determinant > 1e-8)
            residual = predicted-target
            safe = np.where(valid, determinant, 1.)
            update = np.column_stack([(jyy*residual[:, 0]-jxy*residual[:, 1])/safe,
                                      (jxx*residual[:, 1]-jxy*residual[:, 0])/safe])
            estimate -= np.where(valid[:, None], update, 0.)
        valid &= np.isfinite(estimate).all(axis=1) & (np.max(np.abs(estimate-normalized), axis=1) < 1e-5)
    return valid


def collect(audit_path, output, frame_offsets=(-2, -1, 0, 1, 2)):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    audit = json.loads(audit_path.read_text())
    source = json.loads(Path(audit['source_report']).read_text())
    data, inputs = Path(audit['sensor_data']), Path(audit['inputs'])
    if (not frame_offsets or len(frame_offsets) > 15
            or any(not isinstance(x, int) or abs(x) > 100 for x in frame_offsets)):
        raise ValueError('Expected at most 15 bounded integer frame offsets')
    output.mkdir(parents=True, exist_ok=False)
    start = int(source.get('start_timestep', 0))
    origin = np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt')
    frames = sorted({audit['automatic_start_frame']+i for i in frame_offsets
                     if 0 <= audit['automatic_start_frame']+i < int(source['frames'])})
    if not frames:
        raise ValueError('No requested frames lie inside the source sequence')
    records = []
    for frame in frames:
        raw_frame = start+frame
        pose_path = data/'ego_pose'/f'{raw_frame:03d}.txt'
        lidar_path = data/'lidar'/f'{raw_frame:03d}.bin'
        pose = np.linalg.inv(origin)@np.loadtxt(pose_path)
        if lidar_path.stat().st_size % 56:
            raise ValueError('Expected archived Nx14 float32 LiDAR format')
        raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 14)
        xyz = raw[:, 3:6]@pose[:3, :3].T+pose[:3, 3]
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        folder = output/f'frame_{frame:03d}'
        folder.mkdir()
        np.savez_compressed(folder/'lidar_source.npz', xyz=xyz, ground_label=raw[valid, 10])
        for camera in (0, 1, 2):
            image_path = data/'images'/f'{raw_frame:03d}_{camera}.jpg'
            intrinsic_path, extrinsic_path = data/'intrinsics'/f'{camera}.txt', data/'extrinsics'/f'{camera}.txt'
            intr = np.loadtxt(intrinsic_path).flatten()
            if len(intr) != 9:
                raise ValueError('Unsupported camera distortion schema')
            c2s = pose@np.loadtxt(extrinsic_path)@CV_TO_WAYMO
            with Image.open(image_path) as image:
                width, height = image.size
                image.thumbnail((960, 640))
                image.save(folder/f'camera_{camera}_preview.jpg')
            uv, depth = project(xyz, c2s, intr)
            good = (depth > .1) & np.isfinite(uv).all(axis=1)
            good &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            indices = np.flatnonzero(good)
            good[indices] &= projection_on_optical_branch(xyz[indices], c2s, intr, uv[indices], depth[indices])
            np.savez_compressed(folder/f'camera_{camera}_sparse_depth.npz',
                                uv_raw=uv[good].astype(np.float32), depth_z=depth[good].astype(np.float32),
                                camera_to_source=c2s, intrinsic_raw=intr, image_wh=[width, height])
            records.append(dict(frame=frame, camera=camera, reference_kind='sensor',
                                image=fingerprint(image_path), lidar=fingerprint(lidar_path),
                                ego_pose=fingerprint(pose_path), intrinsic=fingerprint(intrinsic_path),
                                extrinsic=fingerprint(extrinsic_path), camera_to_source=c2s.tolist(),
                                sparse_depth_samples=int(good.sum()),
                                dense_depth_available=False, occlusion_validated=False))
    annotations_path = data/'instances/instances_info.json'
    annotations = json.loads(annotations_path.read_text())
    candidates = []
    for identity, item in annotations.items():
        if item['class_name'] != 'Vehicle':
            continue
        ann = item['frame_annotations']
        selected = [i for i, f in enumerate(ann['frame_idx']) if f in [start+x for x in frames]]
        if not selected:
            continue
        candidates.append(dict(instance_key=identity, source_id=item['id'],
                               observations=[dict(raw_frame=ann['frame_idx'][i],
                                                  object_to_source=(np.linalg.inv(origin)@np.asarray(ann['obj_to_world'][i])).tolist(),
                                                  box_lwh=ann['box_size'][i]) for i in selected],
                               library_asset_identity_matched=False, camera_visibility_verified=False))
    report = dict(scene=audit['scene'], sensor_audit=fingerprint(audit_path), views=records,
                  requested_frame_offsets=list(frame_offsets), selected_source_frames=frames,
                  native_vehicle_candidates=candidates, annotation_source=fingerprint(annotations_path),
                  sensor_ground_file=str(audit_path.parent/'sensor_skin.npz'),
                  source_frame='first-ego-relative z-up', raw_sensor_only=True,
                  raw_camera_refinement_applied=False, camera_lidar_synchronization_validated=False,
                  edited_asset_reference_complete=False, n1_edit_certified=False,
                  missing=['3DRealCar identity-linked photos or independent geometry',
                           'dense occlusion-aware local reference for edited vehicle',
                           'post-edit render comparison and trust transaction revalidation'])
    # A bounded alternative: actual source vehicles with identity-linked sensor
    # crops/points. They are NOT the library car and have no inferred visibility.
    anchor = np.asarray(audit['proposed_anchor_source_xyz'])
    candidates.sort(key=lambda item: np.linalg.norm(np.asarray(item['observations'][0]['object_to_source'])[:3, 3]-anchor))
    native_references = []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for index, item in enumerate(candidates[:3]):
        car_folder = output/f'native_vehicle_{index:02d}'
        car_folder.mkdir()
        refs = []
        for annotation in item['observations']:
            frame = annotation['raw_frame']-start
            pose = np.asarray(annotation['object_to_source'])
            size = np.asarray(annotation['box_lwh'])
            with np.load(output/f'frame_{frame:03d}/lidar_source.npz') as points:
                local = (points['xyz']-pose[:3, 3])@pose[:3, :3]
            inside = (np.abs(local) <= size/2).all(axis=1)
            cloud = car_folder/f'frame_{frame:03d}_object_points.npy'
            np.save(cloud, local[inside])
            corners = (signs*size/2)@pose[:3, :3].T+pose[:3, 3]
            for record in [v for v in records if v['frame'] == frame]:
                intrinsic = np.loadtxt(record['intrinsic']['path']).flatten()
                uv, depth = project(corners, np.asarray(record['camera_to_source']), intrinsic)
                if not projection_on_optical_branch(corners, np.asarray(record['camera_to_source']),
                                                    intrinsic, uv, depth).all():
                    continue
                with Image.open(record['image']['path']) as image:
                    low = np.maximum(np.floor(uv.min(axis=0)), 0).astype(int)
                    high = np.minimum(np.ceil(uv.max(axis=0)), image.size).astype(int)
                    if (high-low < 16).any():
                        continue
                    crop = car_folder/f'frame_{frame:03d}_camera_{record["camera"]}.jpg'
                    image.crop((*low, *high)).save(crop)
                refs.append(dict(frame=frame, camera=record['camera'], raw_image=record['image'],
                                 crop=str(crop), crop_xyxy=[*low.tolist(), *high.tolist()],
                                 object_points=str(cloud), lidar_points_in_annotation_box=int(inside.sum()),
                                 foreground_segmentation_verified=False, occlusion_validated=False))
        native_references.append(dict(source_id=item['source_id'], instance_key=item['instance_key'],
                                      observations=item['observations'], sensor_references=refs,
                                      distinct_cameras=len({v['camera'] for v in refs}),
                                      library_identity_matched=False, n1_asset_certified=False))
    report['native_vehicle_sensor_references'] = native_references
    (output/'evidence_manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame-offsets', nargs='+', type=int, default=[-2, -1, 0, 1, 2])
    args = parser.parse_args()
    report = collect(args.sensor_audit, args.output, args.frame_offsets)
    print(json.dumps(dict(scene=report['scene'], sensor_views=len(report['views']),
                          native_vehicle_candidates=len(report['native_vehicle_candidates']),
                          n1_edit_certified=False), indent=2))
