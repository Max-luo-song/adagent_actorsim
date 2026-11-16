"""NumPy geometry matching the installed MMDetection3D legacy LiDAR boxes.

Raw tensor: bottom-center XYZ, dimensions XYZ, clockwise yaw around +Z.
Do not change model tensors or scores to convert a geometry convention.
"""
import numpy as np

BOX_CONVENTION = 'mmdet3d_legacy_lidar_xyz_dims_clockwise_yaw_bottom_center_v1'


def box_rotation_row(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def box_corners(box):
    box = np.asarray(box, dtype=float)
    if (box.ndim != 1 or len(box) < 7 or not np.isfinite(box[:7]).all()
            or np.any(box[3:6] <= 0)):
        raise ValueError('Expected finite nonempty legacy LiDAR box')
    unit = np.stack(np.unravel_index(np.arange(8), [2]*3), axis=1)[[0, 1, 3, 2, 4, 5, 7, 6]]
    points = (unit-[.5, .5, 0.])*box[3:6]
    points[:, :2] = points[:, :2]@box_rotation_row(box[6])
    return points+box[:3]


def box_footprint(box):
    return box_corners(box)[[0, 4, 7, 3], :2]


def points_to_box_xy(points, box):
    box = np.asarray(box, dtype=float)
    return (np.asarray(points)-box[:2])@box_rotation_row(box[6]).T


def assert_box_class_contract(box_class, tensor_factory):
    """Fail at model startup if a dependency upgrade changes corner semantics."""
    samples = np.array([[3., 8., -1., 1.5, 4.3, 1.7, .37],
                        [-4., 2., .2, 2.1, 6.2, 2.5, -.83]])
    actual = box_class(tensor_factory(samples), box_dim=7).corners
    if hasattr(actual, 'detach'):
        actual = actual.detach().cpu().numpy()
    expected = np.stack([box_corners(box) for box in samples])
    if np.shape(actual) != expected.shape or not np.allclose(actual, expected, atol=1e-5, rtol=0):
        raise ValueError('Installed box class differs from shared legacy LiDAR convention')
    return dict(box_convention=BOX_CONVENTION, installed_box_class_verified=True,
                maximum_corner_error_m=float(np.max(np.abs(actual-expected))))
