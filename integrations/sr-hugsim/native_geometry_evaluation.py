"""Post-control evaluation only. Native boxes never enter perception/planning."""
import numpy as np
from scipy.spatial.transform import Rotation

class NativeVehicleEvaluation:
    def __init__(self,state):
        def array(value):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            return np.asarray(value)
        self.trans=array(state['instances_trans']);self.quats=array(state['instances_quats'])
        self.sizes=array(state['instances_size']);self.valid=array(state['instances_fv']).astype(bool)
    def polygons(self,frame):
        rows=[]
        for index in np.flatnonzero(self.valid[frame]):
            l,w,h=self.sizes[index];local=np.array([[-l/2,-w/2,0],[l/2,-w/2,0],[l/2,w/2,0],[-l/2,w/2,0]])
            q=self.quats[frame,index];poly=local@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+self.trans[frame,index]
            rows.append((int(index),poly[:,:2]))
        return rows
    def check(self,source_ego_pose,frame,vehicle):
        from closed172_common import polygon_check
        local=np.array([[-vehicle.half_length_m,-vehicle.half_width_m,0,1],[vehicle.half_length_m,-vehicle.half_width_m,0,1],[vehicle.half_length_m,vehicle.half_width_m,0,1],[-vehicle.half_length_m,vehicle.half_width_m,0,1]])
        ego=(np.asarray(source_ego_pose)@local.T).T[:,:2];rows=[dict(object_index=i,**polygon_check(ego,poly)) for i,poly in self.polygons(frame)]
        nearest=min(rows,key=lambda x:x['gap_m']) if rows else dict(object_index=None,gap_m=None,overlap=False)
        return dict(nearest=nearest,overlap=any(row['overlap'] for row in rows),reference_only=True)
