"""Per-context backend choice; exact caches remain local to immutable geometry."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
pass  # Path is provided by the portable boundary above
import json,numpy as np
from scipy.spatial import cKDTree
from bounded_visibility_geometry import polygon_clearance
from mpc_reuse_v1.cached_geometry import CachedPolygon as ReferenceCachedPolygon,EpistemicRisk as ReferenceRisk,OriginalRisk,STATS
from fast_polygon import PolygonKernel
_DEFAULT=bool(_hugsim_expand(json.loads(Path(__file__).with_name('runtime_policy.json').read_text())).get('compiled_polygon',False));_enabled=ContextVar('compiled_polygon',default=_DEFAULT)
@contextmanager
def use_backend(enabled):
 token=_enabled.set(bool(enabled))
 try:yield
 finally:_enabled.reset(token)
class CompiledCachedPolygon:
    def __init__(self,centers,polygon,margin,tree=None,capacity=4096):
        self._kernel=PolygonKernel(polygon)
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
            values=self._kernel.clearance(states[indices],self.centers,self.margin,tree=self.tree)
            STATS['evaluated_rows']+=len(indices);STATS['evaluation_batches']+=1
            STATS['same_batch_duplicate_rows']+=sum(len(ids)-1 for ids in missing.values())
            for (key,ids),value in zip(missing.items(),values):
                result[ids]=value;self.cache[key]=value
                if len(self.cache)>self.capacity:self.cache.popitem(last=False);STATS['evicted_rows']+=1
        return result

class BulkCompiledPolygon(CompiledCachedPolygon):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.batches=OrderedDict()
    def clearance(self,states):
        states=np.asarray(states)
        if not len(states) or states.dtype!=np.float64:return super().clearance(states)
        relevant=np.ascontiguousarray(states[:,[0,1,2,5,6]])
        if not np.isfinite(relevant).all():return super().clearance(states)
        batch=(relevant.dtype.str,relevant.shape,relevant.tobytes());STATS['requested_rows']+=len(states)
        if batch in self.batches:
            STATS['hit_rows']+=len(states);STATS['batch_hits']+=1;self.batches.move_to_end(batch);return self.batches[batch].copy()
        keys=relevant.view(np.dtype((np.void,40))).ravel().tolist();sentinel=object();values=[self.cache.get(k,sentinel) for k in keys];result=np.empty(len(states));missing={};hits=0
        for i,(k,value) in enumerate(zip(keys,values)):
            if value is sentinel:missing.setdefault(k,[]).append(i)
            else:result[i]=value;hits+=1;self.cache.move_to_end(k)
        STATS['hit_rows']+=hits
        if missing:
            indices=[ids[0] for ids in missing.values()];computed=self._kernel.clearance(states[indices],self.centers,self.margin,tree=self.tree)
            STATS['evaluated_rows']+=len(indices);STATS['evaluation_batches']+=1;STATS['same_batch_duplicate_rows']+=sum(len(ids)-1 for ids in missing.values())
            for (k,ids),value in zip(missing.items(),computed):
                result[ids]=value;self.cache[k]=value
                if len(self.cache)>self.capacity:self.cache.popitem(last=False);STATS['evicted_rows']+=1
        self.batches[batch]=result.copy()
        if len(self.batches)>32:self.batches.popitem(last=False)
        return result

def CachedPolygon(*args,**kwargs):
 from performance_backend import enabled
 cls=(BulkCompiledPolygon if enabled('bulk_pose_cache') else CompiledCachedPolygon) if _enabled.get() else ReferenceCachedPolygon
 return cls(*args,**kwargs)
class EpistemicRisk(ReferenceRisk):
 def __init__(self,*args,**kwargs):
  OriginalRisk.__init__(self,*args,**kwargs)
  self._polygon=CachedPolygon(self.points,self.polygon,self.margin,tree=self.tree)
