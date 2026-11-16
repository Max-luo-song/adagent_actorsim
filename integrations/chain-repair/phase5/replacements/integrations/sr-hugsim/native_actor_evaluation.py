"""Post-hoc reference boxes for reconstructed actors, never planner inputs."""
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation,Slerp


def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)


class NativeActorEvaluation:
    def __init__(self,groups):
        self.groups={}
        for name,state in groups.items():
            t=_array(state['instances_trans']);q=_array(state['instances_quats']);sizes=_array(state['instances_size']);valid=_array(state['instances_fv']).astype(bool)
            if q.ndim==4 and q.shape[2:]==(1,4):q=q[:,:,0,:]
            if t.ndim!=3 or t.shape[2]!=3 or q.shape!=t.shape[:2]+(4,) or sizes.shape!=(t.shape[1],3) or valid.shape!=t.shape[:2] or not np.isfinite(sizes).all() or np.any(sizes<=0):raise ValueError('Invalid native actor reference geometry: '+name)
            self.groups[name]=(t,q,sizes,valid)
    @classmethod
    def from_archive(cls,arrays):
        groups={}
        for key in arrays:
            group,field=key.split('__',1);groups.setdefault(group,{})[field]=arrays[key]
        return cls(groups)
    @lru_cache(maxsize=32)
    def geometry(self,frame):
        rows=[]
        for group,(translations,quats,sizes,valid) in self.groups.items():
            if not np.isfinite(frame) or frame<0 or frame>len(translations)-1:raise ValueError('Actor reference time outside source')
            lo=int(np.floor(frame));hi=int(np.ceil(frame));u=frame-lo
            active=valid[lo] if lo==hi else valid[lo]|valid[hi]
            for index in np.flatnonzero(active):
                if lo==hi or not (valid[lo,index] and valid[hi,index]):
                    k=lo if valid[lo,index] else hi;center=translations[k,index];q=quats[k,index]
                    rotation=Rotation.from_quat(q[[1,2,3,0]])
                else:
                    center=translations[lo,index]*(1-u)+translations[hi,index]*u
                    q=quats[[lo,hi],index][:,[1,2,3,0]];rotation=Slerp([0.,1.],Rotation.from_quat(q))([u])[0]
                l,w,h=sizes[index]
                corners=np.array([[a*l/2,b*w/2,c*h/2] for a in [-1,1] for b in [-1,1] for c in [-1,1]])
                world=rotation.apply(corners)+center;xy=world[:,:2];poly=xy[ConvexHull(xy).vertices]
                rows.append((group,int(index),poly,float(world[:,2].min()),float(world[:,2].max())))
        return rows
    def check(self,pose,frame,vehicle):
        from closed172_common import polygon_check
        pose=np.asarray(pose,dtype=float)
        local=np.array([[a*vehicle.half_length_m,b*vehicle.half_width_m,z,1.] for a in [-1,1] for b in [-1,1] for z in [0.,vehicle.body_height_m]])
        world=(pose@local.T).T[:,:3];xy=world[:,:2];ego=xy[ConvexHull(xy).vertices];low,high=world[:,2].min(),world[:,2].max();rows=[]
        for group,index,poly,z0,z1 in self.geometry(float(frame)):
            if z1<low or z0>high:continue
            rows.append(dict(group=group,object_index=index,**polygon_check(ego,poly)))
        nearest=min(rows,key=lambda x:x['gap_m']) if rows else dict(group=None,object_index=None,gap_m=None,overlap=False)
        return dict(nearest=nearest,overlap=any(x['overlap'] for x in rows),overlaps=[x for x in rows if x['overlap']],reference_only=True,
                    scope='reconstruction actor box proxies with vertical filtering; not mesh contact certification')
