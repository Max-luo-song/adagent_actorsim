"""Bounded exact-pose reuse within one immutable geometry snapshot."""
from collections import OrderedDict,Counter
import numpy as np
from scipy.spatial import cKDTree
from bounded_visibility_geometry import polygon_clearance
from global_repair_v2.boundary_policy import EpistemicRisk as OriginalRisk
STATS=Counter()
class CachedPolygon:
    def __init__(self,centers,polygon,margin,tree=None,capacity=4096):
        if capacity<1:raise ValueError('Positive cache capacity required')
        self.centers=np.array(centers,copy=True).reshape(-1,2)
        self.polygon=np.array(polygon,copy=True);self.margin=margin
        self.tree=cKDTree(self.centers,copy_data=True) if tree is not None and len(self.centers) else None
        self.cache=OrderedDict();self.capacity=capacity
    def clearance(self,states):
        states=np.asarray(states)
        if not len(states):return polygon_clearance(states,self.centers,self.polygon,self.margin,tree=self.tree)
        # These are exactly the fields read by polygon_clearance. Speed and
        # steering are irrelevant; no coordinate/yaw rounding is permitted.
        relevant=np.ascontiguousarray(states[:,[0,1,2,5,6]])
        if not np.isfinite(relevant).all():
            STATS['nonfinite_fallbacks']+=1
            return polygon_clearance(states,self.centers,self.polygon,self.margin,tree=self.tree)
        keys=[(relevant.dtype.str,row.tobytes()) for row in relevant]
        result=np.empty(len(states));missing={}
        for i,key in enumerate(keys):
            STATS['requested_rows']+=1
            if key in self.cache:
                STATS['hit_rows']+=1;self.cache.move_to_end(key);result[i]=self.cache[key]
            else:missing.setdefault(key,[]).append(i)
        if missing:
            indices=[ids[0] for ids in missing.values()]
            values=polygon_clearance(states[indices],self.centers,self.polygon,self.margin,tree=self.tree)
            STATS['evaluated_rows']+=len(indices);STATS['evaluation_batches']+=1
            STATS['same_batch_duplicate_rows']+=sum(len(ids)-1 for ids in missing.values())
            for (key,ids),value in zip(missing.items(),values):
                result[ids]=value;self.cache[key]=value
                if len(self.cache)>self.capacity:self.cache.popitem(last=False);STATS['evicted_rows']+=1
        return result
class EpistemicRisk(OriginalRisk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._polygon=CachedPolygon(self.points,self.polygon,self.margin,tree=self.tree)
    def clearance(self,states):
        if not len(self.points):return np.full(len(states),np.inf)
        expanded=np.c_[states,np.full(len(states),self.vehicle.half_width_m),np.full(len(states),self.vehicle.half_length_m)]
        return self._polygon.clearance(expanded)
