"""CPU conversion of unified RGB/calibration packets to the frozen UniAD contract.

Only observations and ego/sensor metadata enter the model. Managed-object poses,
asset geometry and placement references are kept outside its input packet.
"""
import math
import numpy as np
from waymo_adapter import rigid,SR_TO_HUGSIM as T

B=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
ORDER=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT')


def to_uniad(packet, mount, *, speed_mps, steering_rad, command, scene_token, dt=.1, acceleration_mps2=0., steering_rate_radps=0.):
    from scipy.spatial.transform import Rotation
    if packet.get('schema')!='unified_observation_v1' or tuple(packet['camera_order'])!=ORDER:
        raise ValueError('Unexpected unified observation/camera order')
    if packet.get('training') is not False or packet.get('formal_evaluation_admitted') is not False:
        raise ValueError('Expected an explicit frozen diagnostic observation')
    mount=rigid(mount);ego=rigid(packet['ego_to_world'])
    if not np.allclose(mount[:3,:3],np.eye(3)):raise ValueError('Unsupported virtual LiDAR orientation')
    if not np.isfinite([speed_mps,steering_rad,dt,acceleration_mps2,steering_rate_radps]).all() or speed_mps<0 or dt<=0 or command not in (0,1,2) or not scene_token:
        raise ValueError('Invalid ego/model context')
    native=T@ego@T.T
    virtual=B@native@B.T@mount
    info=dict(command=int(command),cam_rect=np.eye(4),cam_params={},timestamp=float(packet['timestamp_s']),
        source_frame=int(packet['source_frame']),scene_token=str(scene_token),simulation_dt=float(dt),
        ego_pos=native[:3,3].tolist(),ego_rot=Rotation.from_matrix(native[:3,:3]).as_euler('XYZ').tolist(),
        ego_velo=float(speed_mps),ego_steer=float(steering_rad),accelerate=float(acceleration_mps2),steer_rate=float(steering_rate_radps),
        explicit_l2g_r_mat=virtual[:3,:3],explicit_l2g_t=virtual[:3,3],
        model_lidar_frame=dict(virtual_to_ego=mount.tolist()),
        edited_scene=bool(packet['edit_trust']['edited']),edit_trust=dict(packet['edit_trust']),
        diagnostic_closed_loop=True,closed_loop_run=False,control_actions_executed=False,
        formal_evaluation_admitted=False,n1_certified=False,
        ego_kinematics_source='recorded_pose_derivative',input_contract='unified_to_uniad_v1')
    obs={'rgb':{},'depth':{}}
    for name in ORDER:
        view=packet['cameras'][name];rgb=np.asarray(view['rgb']);k=np.asarray(view['K'],dtype=float)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or k.shape!=(3,3):
            raise ValueError('Invalid sensor data')
        height,width=rgb.shape[:2]
        rig=T@rigid(view['camera_to_ego'])
        if not np.allclose(ego@view['camera_to_ego'],view['camera_to_world'],atol=1e-5):
            raise ValueError('Inconsistent camera/ego poses')
        info['cam_params'][name]=dict(intrinsic=dict(W=width,H=height,cx=float(k[0,2]),cy=float(k[1,2]),
            fovx=2*math.atan(width/(2*k[0,0])),fovy=2*math.atan(height/(2*k[1,1]))),
            l2c=np.linalg.inv(rig)@B.T@mount,v2c=np.linalg.inv(rig))
        obs['rgb'][name]=rgb;obs['depth'][name]=np.asarray(view['depth'])
        if 'depth_alpha' in view:obs.setdefault('depth_alpha',{})[name]=np.asarray(view['depth_alpha'])
        if 'depth_variance' in view:obs.setdefault('depth_variance',{})[name]=np.asarray(view['depth_variance'])
        for field in ['surface_depth','surface_valid']:
            if field in view:obs.setdefault(field,{})[name]=np.asarray(view[field])
    encodings={packet['cameras'][name].get('depth_encoding','unknown') for name in ORDER}
    info['depth_encoding']=next(iter(encodings)) if len(encodings)==1 else 'mixed_unknown'
    # Crucially: no managed_objects, calibration asset dimensions or target boxes.
    return obs,info


def placement_reference(request, calibration):
    """Post-hoc reference only; this object must never be merged into model info."""
    pose=rigid(request['asset_to_world'])
    return dict(asset=request['asset'],asset_wlh=calibration['support']['wlh'],
        local_bottom_y=calibration['support']['bottom_y'],
        local_recenter_xz=calibration['support']['raw_recenter_xz'],
        geometry_contract=calibration['geometry_contract'],
        placements=[dict(case='heading_A',asset_to_source=pose.tolist(),asset_to_native=(T@pose).tolist())],
        reference_is_gaussian_support_proxy=True,n1_edit_certified=False,evaluation_admitted=False)
