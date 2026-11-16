"""Swap ground anchors, not camera/object centers; preserve the target identity."""
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

def footprint(pose,length,width):
    corners=np.array([[-length/2,-width/2,0],[length/2,-width/2,0],[length/2,width/2,0],[-length/2,width/2,0]])
    return Polygon((corners@pose[:3,:3].T+pose[:3,3])[:,:2])

def design(samples,actors,up,ego_length=3.,ego_width=1.6,index=3):
    frames=np.array([s['source_frame'] for s in samples]);ego=np.array([s['recorded_ego_to_world'] for s in samples],float)
    car=actors['RigidNodes__instances_trans'][frames,index].astype(float)
    size=actors['RigidNodes__instances_size'][index].astype(float)
    rotation=Rotation.from_quat(actors['RigidNodes__instances_quats'][frames,index][:,[1,2,3,0]]).as_matrix()
    up=np.asarray(up,float);up/=np.linalg.norm(up)
    car_ground=car-rotation[:,:,2]*size[2]/2
    heading=rotation[:,:,0];heading-=np.einsum('ij,j->i',heading,up)[:,None]*up
    heading/=np.linalg.norm(heading,axis=1)[:,None]
    new_ego=np.repeat(np.eye(4)[None],len(frames),axis=0)
    new_ego[:,:3,:3]=np.stack([heading,np.cross(up,heading),np.repeat(up[None],len(frames),axis=0)],axis=-1)
    new_ego[:,:3,3]=car_ground
    new_car=ego[:,:3,3]+rotation[:,:,2]*size[2]/2
    gaps=[];overlaps=[]
    for i,frame in enumerate(frames):
        car_pose=np.eye(4);car_pose[:3,:3]=rotation[i];car_pose[:3,3]=new_car[i]
        a=footprint(new_ego[i],ego_length,ego_width);b=footprint(car_pose,size[0],size[1])
        gaps.append(float(a.distance(b)));overlaps.append(bool(a.intersects(b)))
    audit=dict(schema='ego_actor_position_swap_v1',object_id='native:RigidNodes:3',source_instance_index=5,
        target_identity='o24FGKGBO7aPlRVYkDyU7A',source_frames=frames.tolist(),
        semantics='Positions swapped from the first frame, following exchanged ground-anchor trajectories. No crossing maneuver.',
        height_policy='ego ground origin to target ground anchor; car center to original ego ground plus original half-height',
        minimum_ego_target_footprint_gap_m=float(min(gaps)),ego_target_overlap_frames=[int(f) for f,v in zip(frames,overlaps) if v],
        checks=dict(no_ego_target_overlap=not any(overlaps),
            ego_on_original_target_ground=bool(np.allclose(new_ego[:,:3,3],car_ground)),
            target_on_original_ego_ground=bool(np.allclose(new_car-rotation[:,:,2]*size[2]/2,ego[:,:3,3])),
            rotation_rigid=bool(np.allclose(np.linalg.det(new_ego[:,:3,:3]),1,atol=1e-6))),
        collision_with_background_certified=False,road_compliance_certified=False,closed_loop_driving=False)
    if not all(audit['checks'].values()):raise ValueError('Swap geometry check failed: '+str(audit))
    return new_ego,new_car,audit
