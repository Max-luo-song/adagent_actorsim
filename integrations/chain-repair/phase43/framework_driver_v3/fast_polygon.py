"""Batched pose preparation and an exact SAT reduction with unchanged KD-tree bound.

No point thinning, no coordinate rounding, no geometry/margin changes. Float64
roundoff is checked against the original NumPy implementation before admission.
"""

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

import ctypes,hashlib,json,platform
pass  # Path is provided by the portable boundary above
from collections import Counter
import numpy as np
from bounded_visibility_geometry import polygon_clearance as reference
P=Path(__file__).resolve().parent
record=_hugsim_expand(json.loads((P/'polygon_build_manifest.json').read_text()));path=(P/record['library']).resolve()
if path.parent!=(P/'polygon_build').resolve() or record['machine']!=platform.machine() or record['source_sha256']!=hashlib.sha256((P/'polygon_kernel.cpp').read_bytes()).hexdigest() or record['library_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ImportError('Polygon library/source/platform mismatch')
lib=ctypes.CDLL(str(path));lib.polygon_kernel_abi.restype=ctypes.c_int
if lib.polygon_kernel_abi()!=1:raise ImportError('Unsupported polygon ABI')
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
lib.polygon_csr.argtypes=[D,D,D,D,D,D,I,I,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int,ctypes.c_double,D];lib.polygon_csr.restype=None
STATS=Counter()
def array(x):return np.ascontiguousarray(x,dtype=np.float64)
class PolygonKernel:
 def __init__(self,polygon):
  self.polygon=np.array(polygon,copy=True);edges=np.roll(self.polygon,-1,axis=0)-self.polygon
  self.valid=self.polygon.ndim==2 and self.polygon.shape[1]==2 and len(self.polygon)>=3 and np.isfinite(self.polygon).all() and np.all(np.linalg.norm(edges,axis=1)>0)
  if self.valid:
   normals=np.c_[-edges[:,1],edges[:,0]];self.normals=normals/np.linalg.norm(normals,axis=1)[:,None];self.radius=float(np.linalg.norm(self.polygon,axis=1).max())
 def clearance(self,states,centers,margin,tree=None):
  states=np.asarray(states);centers=np.asarray(centers).reshape(-1,2);n,m=len(states),len(centers)
  if not m:return np.full(n,np.inf)
  if not n:return np.empty(0)
  if not self.valid or states.dtype!=np.float64 or centers.dtype!=np.float64 or states.ndim!=2 or states.shape[1]<7 or not np.isfinite(states[:,[0,1,2,5,6]]).all() or not np.isfinite(centers).all() or not np.isfinite(margin):
   STATS['fallbacks']+=1;return reference(states,centers,self.polygon,margin,tree=tree)
  yaw=states[:,2];right=np.c_[np.cos(yaw),-np.sin(yaw)];forward=np.c_[np.sin(yaw),np.cos(yaw)];a=len(self.normals)+2
  axes=np.empty((n,a,2));axes[:,0]=right;axes[:,1]=forward;axes[:,2:]=self.normals
  body=np.abs(np.matmul(axes,right[:,:,None])[:,:,0])*states[:,5,None]+np.abs(np.matmul(axes,forward[:,:,None])[:,:,0])*states[:,6,None]
  projected=np.matmul(self.polygon,axes.transpose(0,2,1));lo=projected.min(1);hi=projected.max(1)
  offsets=np.zeros(n+1,dtype=np.int64);ids=np.empty(0,dtype=np.int64)
  if tree is not None:
   distances,nearest=tree.query(states[:,:2]);near_delta=np.matmul((centers[nearest]-states[:,:2])[:,None,:],axes.transpose(0,2,1))[:,0,:]
   upper=np.maximum(near_delta+lo-body,-near_delta-hi-body).max(1)
   radii=np.maximum(distances+1e-8,np.sqrt(2.)*(upper+np.hypot(states[:,5],states[:,6])+self.radius)+1e-7)
   neighborhoods=tree.query_ball_point(states[:,:2],radii)
   counts=np.fromiter((len(x) for x in neighborhoods),dtype=np.int64,count=n);offsets[1:]=np.cumsum(counts);ids=np.asarray([j for row in neighborhoods for j in row],dtype=np.int64)
  out=np.empty(n);lib.polygon_csr(array(states[:,:2]),array(axes),array(lo),array(hi),array(body),array(centers),offsets,ids,n,m,a,int(tree is not None),float(margin),out)
  STATS['calls']+=1;STATS['poses']+=n;STATS['candidate_cells']+=len(ids) if tree is not None else n*m
  return out
