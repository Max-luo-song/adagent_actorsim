"""Same frozen neural and external evidence pipeline; typed occupancy export."""

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

import sys,json,types
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path[:0]=[str(HERE),str(CR/'phase29'),str(CR/'phase28/heading_fix_v1'),str(CR/'phase27'),str(ROOT/'waymo23-native-import')]
if __name__=='__main__':
 from runtime_bundle import activate_configured
 activate_configured()
from global_repair_v1 import worker as global_worker
import protocol_branch_worker as worker
origin=global_worker.origin_worker

def serve(model,namespace):
 core=getattr(model,'module',model);original=core.motion_head.forward_test
 def motion(*args,**kwargs):
  result=original(*args,**kwargs)
  count=int(result[1]['track_query'].shape[1])
  current=max(worker.JOB.glob('step_[0-9][0-9][0-9]'))
  worker.atomic_json(current/'occupancy_head_audit.json',dict(motion_query_count=count,no_motion_query=count==0))
  return result
 core.motion_head.forward_test=motion
 try:return global_worker.serve(model,namespace)
 finally:core.motion_head.forward_test=original

def install():
 global_worker.original_serve=worker.serve
 origin.worker=worker
 origin.original_serve=serve
 worker.serve=origin.serve
if __name__=='__main__':
 install();worker.main()
