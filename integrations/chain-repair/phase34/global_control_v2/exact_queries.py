"""Exact snapshot-local road pose queries; no cross-frame or rounded reuse."""
from collections import OrderedDict
import numpy as np
class RoadPoseCache:
 def __init__(self,raw,capacity=4096):self.raw=raw;self.capacity=capacity;self.cache=OrderedDict();self.metrics={'requested_rows':0,'hit_rows':0,'evaluated_rows':0,'evictions':0}
 def __call__(self,states):
  states=np.asarray(states,dtype=float)
  if states.ndim!=2 or states.shape[1]!=5 or not np.isfinite(states).all() or not len(states):return self.raw(states)
  keys=[row.tobytes() for row in np.ascontiguousarray(states[:,:3])];result=np.empty(len(states));missing={}
  for i,key in enumerate(keys):
   self.metrics['requested_rows']+=1
   if key in self.cache:self.metrics['hit_rows']+=1;self.cache.move_to_end(key);result[i]=self.cache[key]
   else:missing.setdefault(key,[]).append(i)
  if missing:
   ids=[indices[0] for indices in missing.values()]
   # Original road implementation batches in groups of 32. Keep that bound
   # and avoid its recursive self.clearance calling this wrapper again.
   values=np.concatenate([self.raw(states[ids[start:start+32]]) for start in range(0,len(ids),32)])
   self.metrics['evaluated_rows']+=len(ids)
   for (key,indices),value in zip(missing.items(),values):
    result[indices]=value;self.cache[key]=value
    if len(self.cache)>self.capacity:self.cache.popitem(last=False);self.metrics['evictions']+=1
  return result
