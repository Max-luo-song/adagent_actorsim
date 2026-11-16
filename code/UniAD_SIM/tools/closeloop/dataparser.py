import cv2
import mmcv
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math
from projects.mmdet3d_plugin.uniad.input_geometry import camera_projection, intrinsic_matrix, resize_plan

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
OPENCV2LIDAR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_intrinsic(intrin_dict):
    return intrinsic_matrix(intrin_dict)

def parse_raw(raw_data, cameras, img_norm_cfg, history, *, perception_preview=False, resize_mode='stretch'):
    obs, info = raw_data
    from projects.mmdet3d_plugin.uniad.edit_trust import assert_edit_evaluable
    if not perception_preview:
        assert_edit_evaluable(info)
    imu_rot = np.array([0,0,0], dtype=np.float32)
    imu_rot[2] = -info['ego_rot'][1]
    lidar_rot = np.array([0,0,0], dtype=np.float32)
    lidar_rot[2] = info['ego_rot'][1]
    
    imgs_shape = []
    imgs = []
    raw_imgs = []
    l2imgs, intrinsics = [], []
    
    for cam in cameras:
        im = obs['rgb'][cam]
        resized_K, lidar2img = camera_projection(info['cam_params'][cam], im.shape, (1600, 900), resize_mode)
        plan = resize_plan((im.shape[1], im.shape[0]), (1600, 900), resize_mode)
        im = cv2.resize(im, plan['resized_wh'])
        im_norm = mmcv.imnormalize(im, img_norm_cfg['mean'], img_norm_cfg['std'], img_norm_cfg['to_rgb'])
        left, top, right, bottom = plan['pad_ltrb']
        im_norm = np.pad(im_norm, ((top, bottom), (left, right), (0, 0)), constant_values=0)
        imgs_shape.append(list(im_norm.shape))
        im_norm = torch.from_numpy(im_norm).permute(2, 0, 1)
        imgs.append(im_norm)
        raw_imgs.append(im)
        l2imgs.append(lidar2img)
        intrinsics.append(resized_K)

    imgs = torch.stack(imgs)[None, ...]
    imgs = imgs.cuda()
    timestamp = info['timestamp']
    
    f2w = np.eye(4)
    f2w[:3, :3] = SCR.from_euler('XYZ', info['ego_rot']).as_matrix()
    f2w[:3, 3] = info['ego_pos']

    l2g_r_mat = SCR.from_euler('XYZ', lidar_rot).as_matrix()
    l2g_t = OPENCV2LIDAR @ f2w[:3, 3]

    can_bus = np.zeros(18)
    can_bus[:3] = OPENCV2IMU @ f2w[:3, 3]
    can_bus[2] = 0
    v_quat = SCR.from_euler('XYZ', imu_rot).as_quat()[[3,0,1,2]]
    can_bus[3:7] = v_quat
    
    forward_acc = info['accelerate']
    forward_velo = info['ego_velo']
    steer = -info['ego_steer']
    # accel
    # can_bus[7:10] = np.array([forward_acc * np.cos(steer), forward_acc * np.sin(steer), 9.8])
    # can_bus[7:10] = np.array([forward_acc, 0, 0])
    # if info['command'] != 2:
    #     can_bus[7:10] = np.array([0, 0, 9.8])
    # can_bus[7:10] = np.array([0, 0, 9.8])
    # rotation_rate
    can_bus[10:13] = np.array([0, 0, -info['steer_rate']])
    # vel
    can_bus[13:16] = np.array([forward_velo, 0, 0])
    
    yaw = imu_rot[2]
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180
    
    
    img_metas = {
        'diagnostic_preview_only': bool(perception_preview),
        'edit_trust': info.get('edit_trust'),
        'scene_token': '062',
        'can_bus': can_bus,
        'img_shape': imgs_shape,
        'lidar2img': l2imgs,
        'intrinsics': intrinsics,
    }
    img_metas = [[img_metas]]
    data = {
        'img': imgs, 
        'img_metas': img_metas,
        'l2g_t': torch.tensor(l2g_t).float(),
        'l2g_r_mat': torch.tensor(l2g_r_mat).float(),
        'timestamp': timestamp,
        'command': [torch.tensor([info['command']]).cuda()], # 0 right, 1 left, 2 straight
        'raw_imgs': raw_imgs,
    }

    return data

