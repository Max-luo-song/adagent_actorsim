"""Shared metric asset geometry for placement, projection and offline evaluation.

Local axes: X longitudinal (signed front unverified), Y down, Z lateral.
Dimensions: width, length, height. Gaussian support is not a physical mesh.
"""
import numpy as np

SCHEMA = 'asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1'


def local_corners(wlh, bottom_y):
    dims = np.asarray(wlh, dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0) or not np.isfinite(bottom_y):
        raise ValueError('Expected positive finite asset width/length/height and bottom')
    width, length, height = dims
    return np.array([[x, y, z] for x in (-length/2, length/2)
                     for y in (bottom_y-height, bottom_y) for z in (-width/2, width/2)])


def validate_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0,0,0,1], atol=1e-8, rtol=0)
            or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3,:3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Asset placement must be a finite rigid proper transform')
    return pose


def source_corners(asset):
    declared = asset.get('geometry_contract', {}).get('schema', SCHEMA)
    if declared != SCHEMA:
        raise ValueError('Unsupported asset geometry convention')
    pose = validate_pose(asset['placements'][0]['asset_to_source'])
    local = local_corners(asset['asset_wlh'], asset['local_bottom_y'])
    return local@pose[:3,:3].T+pose[:3,3]


def bottom_polygon(wlh, bottom_y):
    # Preserve the historical evaluator polygon order exactly.
    width, length, _ = np.asarray(wlh, dtype=float)
    local_corners(wlh, bottom_y)
    return np.array([[x,bottom_y,z] for x,z in
                     [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]])


def declared_contract(wlh, bottom_y):
    local_corners(wlh,bottom_y)
    return dict(schema=SCHEMA, axes=dict(x='longitudinal',y='down',z='lateral'),
                dimension_order=['width','length','height'],unit='meter',
                wlh=list(map(float,wlh)),bottom_y=float(bottom_y),
                signed_front_verified=False,physical_dimensions_independently_verified=False,
                real_wheel_contacts_verified=False)
