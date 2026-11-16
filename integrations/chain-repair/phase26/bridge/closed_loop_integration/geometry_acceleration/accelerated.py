"""Snapshot-local acceleration only: original road kernel + compiled polygon gap.

No search/constraint/margin changes, approximate memo keys or cross-frame reuse.
Install in the dedicated driver process after frozen configure(), before geometry
objects are built. Source files and their manifests stay byte-for-byte intact.
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

import ctypes
import hashlib
import importlib
import inspect
import json
pass  # Path is provided by the portable boundary above
import textwrap
import numpy as np
HERE=Path(__file__).resolve().parent
STATS=dict(polygon_calls=0,polygon_rows=0,polygon_fallbacks=0)
_INSTALLED=[]
_LIB=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def library():
 global _LIB
 if _LIB is not None:return _LIB
 record=_hugsim_expand(json.loads((HERE/'build_manifest.json').read_text()));path=Path(record['library']).resolve()
 if path.parent!=HERE/'build' or sha(path)!=record['library_sha256'] or sha(HERE/'polygon.cpp')!=record['source_sha256']:raise RuntimeError('Geometry binary/source fingerprint mismatch')
 lib=ctypes.CDLL(str(path));lib.edited_polygon_abi.restype=ctypes.c_int
 if lib.edited_polygon_abi()!=1:raise RuntimeError('Geometry ABI mismatch')
 D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
 lib.projected_polygon_gap.argtypes=[D,D,D,D,D,D,I,I,ctypes.c_int64,ctypes.c_int64,ctypes.c_double,D]
 lib.projected_polygon_gap.restype=None;_LIB=lib;return lib

def polygon_function(original):
 def polygon_clearance(states,centers,polygon,margin,tree=None):
  STATS['polygon_calls']+=1
  states=np.asarray(states);centers=np.asarray(centers).reshape(-1,2);polygon=np.asarray(polygon)
  if not len(centers):return np.full(len(states),np.inf)
  if not len(states):return np.asarray([])
  # Preserve original validation/failure semantics outside the proven domain.
  if states.ndim!=2 or states.shape[1]<7 or polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not all(np.isfinite(x).all() for x in (states,centers,polygon)) or not np.isfinite(margin):
   STATS['polygon_fallbacks']+=1;return original(states,centers,polygon,margin,tree=tree)
  # Scalar trig/matmul preparation matches the reference arithmetic exactly.
  # Move only the point/axis scan into C; batch KD queries to reduce dispatch.
  edges=np.roll(polygon,-1,axis=0)-polygon
  normals=np.c_[-edges[:,1],edges[:,0]]
  if np.any(np.linalg.norm(normals,axis=1)==0):
   STATS['polygon_fallbacks']+=1;return original(states,centers,polygon,margin,tree=tree)
  normals/=np.linalg.norm(normals,axis=1)[:,None]
  cell_radius=float(np.linalg.norm(polygon,axis=1).max())
  if tree is not None:distances,nearest=tree.query(states[:,:2])
  all_axes=[];all_low=[];all_high=[];all_body=[];radii=[]
  for row,state in enumerate(states):
   yaw=state[2];right=np.array([np.cos(yaw),-np.sin(yaw)]);forward=np.array([np.sin(yaw),np.cos(yaw)])
   axes=np.vstack([right,forward,normals]);body=np.abs(axes@right)*state[5]+np.abs(axes@forward)*state[6]
   projected=polygon@axes.T;low=projected.min(0);high=projected.max(0)
   all_axes.append(axes);all_low.append(low);all_high.append(high);all_body.append(body)
   if tree is not None:
    near_delta=(centers[int(nearest[row])]-state[:2])@axes.T
    upper=float(np.maximum(near_delta+low-body,-near_delta-high-body).max())
    radii.append(max(float(distances[row])+1e-8,np.sqrt(2.)*(upper+np.hypot(state[5],state[6])+cell_radius)+1e-7))
  if tree is not None:groups=tree.query_ball_point(states[:,:2],np.asarray(radii))
  else:groups=[np.arange(len(centers))]*len(states)
  starts=np.r_[0,np.cumsum([len(ids) for ids in groups])].astype(np.int64)
  ids=np.ascontiguousarray(np.concatenate(groups),dtype=np.int64)
  if not len(ids) or np.any(ids<0) or np.any(ids>=len(centers)):
   STATS['polygon_fallbacks']+=1;return original(states,centers,polygon,margin,tree=tree)
  def arr(value):return np.ascontiguousarray(value,dtype=np.float64)
  result=np.empty(len(states),dtype=np.float64)
  library().projected_polygon_gap(arr(states[:,:2]),arr(centers),arr(all_axes),arr(all_low),arr(all_high),arr(all_body),starts,ids,len(states),len(normals)+2,float(margin),result)
  STATS['polygon_rows']+=len(states)
  return result
 return polygon_clearance

def install():
 if _INSTALLED:raise RuntimeError('Geometry acceleration already installed')
 library()
 # Replace only this method's local import, not planning_guard globally.
 from drivable_area import DrivableArea
 old=DrivableArea.clearance
 source=textwrap.dedent(inspect.getsource(old));needle='from planning_guard import rectangle_separation'
 if source.count(needle)!=1:raise RuntimeError('Unreviewed road method')
 scope=dict(old.__globals__)
 exec(compile(source.replace(needle,'from mpc_native_v1.native import rectangle_separation'),str(HERE/'road_import_adapter.py'),'exec'),scope)
 patched=scope['clearance'];DrivableArea.clearance=patched;_INSTALLED.append((DrivableArea,'clearance',old,patched))
 try:
  modules=[importlib.import_module(name) for name in ('bounded_visibility_geometry','mpc_reuse_v1.cached_geometry','empty_bounded_geometry','empty_native_temporal')]
  for module in modules:
   old=getattr(module,'polygon_clearance',None)
   if old is not None:
    patched=polygon_function(old);setattr(module,'polygon_clearance',patched);_INSTALLED.append((module,'polygon_clearance',old,patched))
 except BaseException:
  uninstall();raise
 return audit()

def uninstall():
 for owner,name,old,patched in reversed(_INSTALLED):
  if getattr(owner,name) is not patched:raise RuntimeError('Accelerator binding changed unexpectedly')
  setattr(owner,name,old)
 _INSTALLED.clear()

def audit():
 record=_hugsim_expand(json.loads((HERE/'build_manifest.json').read_text()))
 files=[HERE/'accelerated.py',HERE/'polygon.cpp',HERE/'build.py',HERE/'build_manifest.json',Path(record['library'])]
 return dict(enabled=bool(_INSTALLED),scope='road_SAT_and_polygon_point_axis_scan',bindings=[dict(owner=getattr(owner,'__name__',str(owner)),attribute=name) for owner,name,old,patched in _INSTALLED],implementation={str(p):sha(p) for p in files},statistics=dict(STATS),search_policy_changed=False,constraints_changed=False,source_files_modified=False)
