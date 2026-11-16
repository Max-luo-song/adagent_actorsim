"""Unified full feedback entry, no recorded control/observation prefix."""

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

import json,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(HERE.parent),str(CR/'phase28'),str(ROOT/'waymo23-native-import')]
import heading_scene_run_v1 as harness
from global_repair_v1.driver import ContractObstacleMemory,validate_guard_input

def main():
    if sys.argv[1:]==['--check-imports']:
        print('Global driver imports ready');return
    request=_hugsim_expand(json.loads(Path(sys.argv[1]).read_text()))
    if request['mode'] not in ('record','branch','replay-branch'):raise ValueError('Unknown global driver mode')
    harness.WORKER=HERE/'worker.py';harness.ObstacleMemory=ContractObstacleMemory
    harness._PATCH_MODULE_PATHS.extend(f for f in HERE.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.sh'))
    original=harness.optimize
    def optimize(reference,occupancy,info,*args,**kwargs):
        validate_guard_input(kwargs['reliability'],info)
        return original(reference,occupancy,info,*args,**kwargs)
    harness.optimize=optimize
    loop=harness._control_loop
    def control(ctx,output,session,report,*args,**kwargs):
        report.update(experiment='unified_global_evidence_repair',neural_query_retirement=False,recorded_prefix_used=False,
            generalization_certified=False,global_repair_complete=False,perception_coverage='existing three-camera rig unless a declared auxiliary sensor contract is supplied')
        return loop(ctx,output,session,report,*args,**kwargs)
    harness._control_loop=control
    if request['mode']=='record':harness.run_record(request)
    else:harness.run_branch(request,replay=request['mode']=='replay-branch')
if __name__=='__main__':main()
