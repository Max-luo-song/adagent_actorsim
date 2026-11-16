"""One resident frozen model, isolated diagnostic cases, no vehicle execution."""

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

import sys,importlib.util
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path[:0]=[str(LOCAL.parent/'light-cutin-diagnostic'),str(ROOT/'waymo23-native-import')]
_spec=importlib.util.spec_from_file_location('_uniad_speed_mechanism_hooks_v2',LOCAL/'hooks.py')
_hooks=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_hooks)
install=_hooks.install
import worker_trace as trace
from common import read,write

def serve(model,namespace):
    core=getattr(model,'module',model);protocol=trace.frozen_worker.worker
    cleanup=install(core,lambda:max(protocol.JOB.glob('step_[0-9][0-9][0-9]')),protocol,read,write)
    try:return trace.trace_serve(model,namespace)
    finally:cleanup()

if __name__=='__main__':
    trace.frozen_worker.serve=serve;trace.frozen_worker.install();trace.frozen_worker.worker.main()
