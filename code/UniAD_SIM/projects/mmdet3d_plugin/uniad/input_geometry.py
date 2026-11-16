"""Scene-independent pinhole calibration at the actual model image resolution."""
import math

import numpy as np


def resize_plan(source_wh, target_wh=(1600, 900), mode='stretch'):
    width, height = source_wh
    target_w, target_h = target_wh
    if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (*source_wh, *target_wh)):
        raise ValueError('Image dimensions must be positive integers')
    if mode == 'stretch':
        resized_w, resized_h = target_w, target_h
    elif mode == 'letterbox':
        scale = min(target_w/width, target_h/height)
        resized_w = min(target_w, max(1, round(width*scale)))
        resized_h = min(target_h, max(1, round(height*scale)))
    else:
        raise ValueError('Unknown image resize mode')
    left, top = (target_w-resized_w)//2, (target_h-resized_h)//2
    affine = np.eye(4)
    affine[0, 0], affine[1, 1] = resized_w/width, resized_h/height
    affine[0, 2], affine[1, 2] = left, top
    return dict(resized_wh=(resized_w, resized_h),
                pad_ltrb=(left, top, target_w-resized_w-left, target_h-resized_h-top), affine=affine)


def intrinsic_matrix(intrinsic, target_wh=None, image_shape=None, resize_mode='stretch'):
    width, height = intrinsic['W'], intrinsic['H']
    values = [width, height, intrinsic['cx'], intrinsic['cy'], intrinsic['fovx'], intrinsic['fovy']]
    if (not np.isfinite(values).all() or width <= 0 or height <= 0
            or not 0 < intrinsic['fovx'] < math.pi or not 0 < intrinsic['fovy'] < math.pi):
        raise ValueError('Invalid source camera calibration')
    if image_shape is not None and tuple(image_shape[:2]) != (height, width):
        raise ValueError('Image/calibration size mismatch')
    target_w, target_h = target_wh or (width, height)
    if not np.isfinite([target_w, target_h]).all() or min(target_w, target_h) <= 0:
        raise ValueError('Invalid model image size')
    plan = resize_plan((width, height), (target_w, target_h), resize_mode)
    k = np.eye(4)
    k[0, 0] = width/(2*math.tan(intrinsic['fovx']/2))
    k[1, 1] = height/(2*math.tan(intrinsic['fovy']/2))
    k[0, 2], k[1, 2] = intrinsic['cx'], intrinsic['cy']
    return plan['affine']@k


def camera_projection(camera, image_shape, target_wh=(1600, 900), resize_mode='stretch'):
    l2c = np.asarray(camera['l2c'], dtype=float)
    if (l2c.shape != (4, 4) or not np.isfinite(l2c).all()
            or not np.allclose(l2c[3], [0, 0, 0, 1])
            or not np.allclose(l2c[:3, :3].T@l2c[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(l2c[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Invalid lidar-to-camera transform')
    k = intrinsic_matrix(camera['intrinsic'], target_wh, image_shape, resize_mode)
    return k, k@l2c


def model_input_audit(data):
    """Small JSON-safe snapshot of geometry immediately before model forward."""
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value).tolist()
    meta = data['img_metas'][0][0]
    return dict(stage='before_model_forward', tensor_shape=list(data['img'].shape),
                image_shapes=meta['img_shape'], intrinsic_matrices=array(meta['intrinsics']),
                lidar_to_image_matrices=array(meta['lidar2img']), can_bus=array(meta['can_bus']),
                l2g_rotation_row=array(data['l2g_r_mat']), l2g_translation=array(data['l2g_t']),
                timestamp=float(data['timestamp']), scene_token=meta['scene_token'])
