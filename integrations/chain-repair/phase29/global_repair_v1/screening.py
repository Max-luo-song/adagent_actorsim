"""Fixed-input admission checks; no physical-ground-truth claims."""
import numpy as np
from input_geometry import intrinsic_matrix

def center_visible(box,info):
    point=np.r_[np.asarray(box[:3])+[0,0,box[5]/2],1.]
    for camera in info.get('cam_params',{}).values():
        cp=np.asarray(camera['l2c'])@point;k=intrinsic_matrix(camera['intrinsic']);uv=k[:3,:3]@cp[:3]
        if cp[2]>.1 and 0<=uv[0]/cp[2]<camera['intrinsic']['W'] and 0<=uv[1]/cp[2]<camera['intrinsic']['H']:return True
    return False

def same_obstacle(original,filtered):
    if filtered is None or original.keys()!=filtered.keys():return False
    for key,value in original.items():
        if key in ('box','path_clearance_m'):
            if np.shape(value)!=np.shape(filtered[key]) or not np.allclose(value,filtered[key],rtol=0,atol=1e-7):return False
        elif value!=filtered[key]:return False
    return True

def retained_reference_check(original,filtered,info):
    kept={o['candidate_index']:o for o in filtered['obstacles']};references=0;misses=[]
    for obstacle in original['obstacles']:
        if obstacle['state']=='network_retained' and center_visible(obstacle['box'],info):
            references+=1
            if not same_obstacle(obstacle,kept.get(obstacle['candidate_index'])):misses.append(obstacle['candidate_index'])
    return dict(reference_observations=references,changed_or_removed=misses)
