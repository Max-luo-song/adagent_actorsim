"""Metadata-only native RGB diagnostics; source labels never enter inference packets."""
import json
import math
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from build_source_sequence import training_lidar_mount,recorded_kinematics
from runtime_configuration import load_vehicle_config

A=np.array([[0.,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
T=np.array([[0.,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
CV_TO_WAYMO=np.array([[0.,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])


def virtual_geometry(relative,camera_to_ego_cv,mount):
    pose=A@relative@np.linalg.inv(A)@mount
    lidar_to_camera=np.linalg.inv(camera_to_ego_cv)@np.linalg.inv(A)@mount
    return pose,lidar_to_camera


def build(source_path,output):
    import cv2
    source_path=Path(source_path);source=json.loads(source_path.read_text());data=Path(source['inputs']);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    poses=np.stack([np.loadtxt(p) for p in sorted((data/'ego_pose').glob('*.txt'))]);relative=np.linalg.inv(poses[0])[None]@poses
    fps=float(source['pose_fps_assumption']);vehicle=load_vehicle_config()
    if len(poses)<11:raise ValueError('Native perception diagnostic requires at least 11 frames')
    # Perception-only evaluation also includes stationary scenes. No driving
    # admission threshold is changed and no object annotations select frames.
    selected=source.get('selected_frames',[0,5,10])
    if (not isinstance(selected,list) or not selected or len(selected)>100
            or any(not isinstance(f,int) or isinstance(f,bool) or not 0<=f<len(poses) for f in selected)
            or selected!=sorted(set(selected))):raise ValueError('Invalid declared native frame selection')
    mount=training_lidar_mount(Path(__file__).resolve().parents[1]/'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    report=dict(scene=source['scene'],source_report=str(source_path.resolve()),frames=[],scene_edits=[],edit_trust=dict(edited=False),
                diagnostic_preview_only=True,perception_preview_requested=True,input_domain='native_RGB_undistorted',
                timestamp_frequency_independently_validated=False,selection_rule='first three frames at fixed stride5; includes stationary perception-only scenes; no labels',
                ground_truth_in_model_input=False)
    bindings=[]
    from perception_run_binding import file_fingerprint
    for frame in selected:
        obs=dict(rgb={});info=dict(timestamp=frame/fps,scene_token='native-'+str(source['scene']),command=2,accelerate=0.,steer_rate=0.,cam_params={})
        speed,steer,command=recorded_kinematics(relative,frame,fps,vehicle.wheelbase_m);native=T@relative[frame]@np.linalg.inv(T)
        pose=None
        for cid,name in ((0,'CAM_FRONT'),(1,'CAM_FRONT_LEFT'),(2,'CAM_FRONT_RIGHT')):
            image_path=data/'images'/f'{frame:03d}_{cid}.jpg';image=np.asarray(Image.open(image_path).convert('RGB'));h,w=image.shape[:2]
            parameters=np.loadtxt(data/'intrinsics'/f'{cid}.txt').reshape(-1);k=np.array([[parameters[0],0,parameters[2]],[0,parameters[1],parameters[3]],[0,0,1.]])
            # Full-size pinhole rectification; black invalid edges remain explicit.
            rectified=cv2.undistort(image,k,parameters[4:9],None,k)
            scale=float(source.get('image_scale',1.))
            if not 0<scale<=1:raise ValueError('Invalid native image scale')
            if scale!=1:
                resized_w,resized_h=round(w*scale),round(h*scale)
                rectified=cv2.resize(rectified,(resized_w,resized_h))
                k=np.diag([resized_w/w,resized_h/h,1.])@k;w,h=resized_w,resized_h
            obs['rgb'][name]=rectified
            camera=np.loadtxt(data/'extrinsics'/f'{cid}.txt')@CV_TO_WAYMO
            pose,l2c=virtual_geometry(relative[frame],camera,mount)
            info['cam_params'][name]=dict(intrinsic=dict(W=w,H=h,cx=float(k[0,2]),cy=float(k[1,2]),fovx=2*math.atan(w/(2*k[0,0])),fovy=2*math.atan(h/(2*k[1,1]))),l2c=l2c)
            bindings.append(file_fingerprint(image_path))
        info.update(ego_pos=native[:3,3].tolist(),ego_rot=Rotation.from_matrix(native[:3,:3]).as_euler('XYZ').tolist(),
                    explicit_l2g_r_mat=pose[:3,:3],explicit_l2g_t=pose[:3,3],ego_velo=speed,ego_steer=steer,command=command)
        packet=output/f'frame_{frame:03d}.pkl'
        with packet.open('wb') as stream:pickle.dump((obs,info),stream,protocol=4)
        report['frames'].append(dict(frame=frame,packet=str(packet.resolve()),timestamp=frame/fps,scene_token=info['scene_token']))
    report['native_image_bindings']=bindings
    (output/'sequence_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
