
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

pass  # Path is provided by the portable boundary above
import json,sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'bridge/closed_loop_integration')]
from full_scene_contract import verify,bindings
verify(HERE)
from run_edited_closed_loop import preflight
_,_,_,audit=preflight(HERE/'task.json')
paths=bindings(HERE)
assert audit['validation_scope']==dict(name='full_scene_19p8s_v1',steps=198,max_wall_seconds=43200)
assert audit['runtime_road_bundle']['search_migration']['own_paired_cpu_acceptance'] is True
print(json.dumps(dict(passed=True,driver=audit['driver_profile'],scope=audit['validation_scope'],expected_observations=199,adapter_snapshot=audit['adapter_snapshot'],frozen_bindings=paths,bbox_required=True,proxy_avoidance_required_for_success=True,gpu_executed=False),indent=2))
