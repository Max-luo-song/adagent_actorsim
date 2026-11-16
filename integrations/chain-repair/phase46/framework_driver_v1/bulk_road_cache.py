"""Whole-query exact hits and aggregated bookkeeping; no spatial rounding."""
from collections import OrderedDict
import numpy as np
from exact_queries import RoadPoseCache
class BulkRoadPoseCache(RoadPoseCache):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.batches=OrderedDict();self.metrics['batch_hits']=0
 def __call__(self,states):
  states=np.asarray(states,dtype=float)
  if states.ndim!=2 or states.shape[1]!=5 or not np.isfinite(states).all() or not len(states):return self.raw(states)
  relevant=np.ascontiguousarray(states[:,:3]);batch=(relevant.shape,relevant.tobytes());self.metrics['requested_rows']+=len(states)
  if batch in self.batches:
   self.metrics['hit_rows']+=len(states);self.metrics['batch_hits']+=1;self.batches.move_to_end(batch);return self.batches[batch].copy()
  keys=relevant.view(np.dtype((np.void,24))).ravel().tolist();sentinel=object();values=[self.cache.get(k,sentinel) for k in keys];result=np.empty(len(states));missing={};hits=0
  for i,(k,value) in enumerate(zip(keys,values)):
   if value is sentinel:missing.setdefault(k,[]).append(i)
   else:result[i]=value;hits+=1;self.cache.move_to_end(k)
  self.metrics['hit_rows']+=hits
  if missing:
   ids=[indices[0] for indices in missing.values()];computed=np.concatenate([self.raw(states[ids[start:start+32]]) for start in range(0,len(ids),32)]);self.metrics['evaluated_rows']+=len(ids)
   for (k,indices),value in zip(missing.items(),computed):
    result[indices]=value;self.cache[k]=value
    if len(self.cache)>self.capacity:self.cache.popitem(last=False);self.metrics['evictions']+=1
  self.batches[batch]=result.copy()
  if len(self.batches)>32:self.batches.popitem(last=False)
  return result
