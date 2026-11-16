"""Same frozen worker/model; one explicit native-collision policy intervention."""

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

import importlib.util,sys
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path.insert(0,str(LOCAL.parent/'light-cutin-diagnostic'))
import worker_trace as trace
spec=importlib.util.spec_from_file_location('_cutin_native_collision_policy',LOCAL/'native_collision.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
from common import read,write

def serve(model,namespace):
 core=getattr(model,'module',model);protocol=trace.frozen_worker.worker;cleanup=policy.install(core,protocol,read,write);original_write=protocol.atomic_json
 def announce(path,data):
  if Path(path)==protocol.JOB/'ready.json':data=dict(data,native_collision_postprocessor_policy=policy.POLICY)
  return original_write(path,data)
 protocol.atomic_json=announce
 try:return trace.trace_serve(model,namespace)
 finally:protocol.atomic_json=original_write;cleanup()
if __name__=='__main__':
 trace.frozen_worker.serve=serve;trace.frozen_worker.install();trace.frozen_worker.worker.main()
