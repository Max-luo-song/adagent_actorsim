import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math
from navsim.common.dataclasses import AgentInput, EgoStatus, Cameras, Camera, Lidar

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
OPENCV2LIDAR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
nusc_cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
nuplan_cameras = ["cam_f0", "cam_r0", "cam_l0"]

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_intrinsic(intrin_dict):
    fovx, fovy = intrin_dict['fovx'], intrin_dict['fovy']
    h, w = intrin_dict['H'], intrin_dict['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrin_dict['cx'], intrin_dict['cy']
    return K

def depth_to_lidar(obs, info, stride=4):
    """Back-project front-camera expected depth into sparse lidar-frame points."""
    cams = info["cam_params"]
    v2front = np.asarray(cams["CAM_FRONT"]["v2c"], dtype=np.float32)
    cam_rect = np.asarray(info.get("cam_rect", np.eye(4)), dtype=np.float32)
    all_points = []

    for name in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"):
        depth = np.asarray(obs["depth"][name], dtype=np.float32)
        h, w = depth.shape
        intr = cams[name]["intrinsic"]
        fx = fov2focal(intr["fovx"], w)
        fy = fov2focal(intr["fovy"], h)

        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[::stride, ::stride]
        valid = np.isfinite(d) & (d > 0.1) & (d < 100.0)
        if not np.any(valid):
            continue

        cam_pts = np.stack([
            (uu - intr["cx"]) * d / fx,
            (vv - intr["cy"]) * d / fy,
            d,
            np.ones_like(d),
        ], axis=-1)[valid]

        cam_to_front = (
            v2front
            @ np.linalg.inv(np.asarray(cams[name]["v2c"], dtype=np.float32))
            @ cam_rect
        )
        front_pts = (cam_to_front @ cam_pts.T).T[:, :3]
        lidar_pts = (OPENCV2IMU @ front_pts.T).T

        inside = (
            (np.abs(lidar_pts[:, 0]) < 32.0)
            & (np.abs(lidar_pts[:, 1]) < 32.0)
            & (lidar_pts[:, 2] < 100.0)
        )
        all_points.append(lidar_pts[inside])

    points = (
        np.concatenate(all_points, axis=0)
        if all_points else np.zeros((0, 3), dtype=np.float32)
    )
    cloud = np.zeros((6, points.shape[0]), dtype=np.float32)
    cloud[:3] = points.T
    return cloud


def parse_raw(raw_data):
    obs, info = raw_data
    
    imgs = {}
    raw_imgs = {}
    
    for cam in nusc_cameras:
        im = obs['rgb'][cam]
        resize_im = cv2.resize(im, (1920, 1080))
        raw_imgs[cam] = im
        imgs[cam] = resize_im

    velo, acc = np.zeros(2), np.zeros(2)
    # HUGSIM route IDs: 0=unknown, 1=left, 2=straight.
    # NAVSIM command slots: 0=left, 1=straight, 2=right, 3=unknown.
    command_map = {0: 3, 1: 0, 2: 1}
    hugsim_command = int(info['command'])
    if hugsim_command not in command_map:
        raise ValueError(f"Invalid HUGSIM driving command: {hugsim_command}")
    command = np.zeros(4, dtype=np.float32)
    command[command_map[hugsim_command]] = 1.0
    
    ego_pose = OPENCV2IMU @ info['ego_pos']
    # yaw = -info['ego_rot'][1]
    yaw = -info['ego_rot'][1]
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    velo[0] = forward_velo * np.cos(yaw)
    velo[1] = forward_velo * np.sin(yaw)
    acc[0] = forward_acc * np.cos(yaw)
    acc[1] = forward_acc * np.sin(yaw)
    ego_status = EgoStatus(ego_pose, velo, acc, command)
    
    cameras_dict = {}
    for nusc_cam, nuplan_cam in zip(nusc_cameras, nuplan_cameras):
        cameras_dict[nuplan_cam] = Camera(image=imgs[nusc_cam])
        
    cameras = Cameras(*([None] * 8))
    for cam_name, cam in cameras_dict.items():
        setattr(cameras, cam_name, cam)
        
    lidar = Lidar(depth_to_lidar(obs, info))
    
    agent_input = AgentInput([ego_status], [cameras], [lidar])
    
    data = {
        'input': agent_input,
        'raw_imgs': raw_imgs,
    }

    return data

