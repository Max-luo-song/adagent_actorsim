"""Driver-only bounded-unknown policy; use the already GPU-validated frozen NN worker."""

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
import depth_sensor_map
from global_repair_v1.driver import ContractObstacleMemory,validate_guard_input
from global_repair_v2.boundary_policy import BoundedDepthMemory
from global_repair_v2.controller import optimize as boundary_optimize

def configure():
    harness.WORKER=HERE.parent/'global_repair_v1/worker.py'
    harness.ObstacleMemory=ContractObstacleMemory
    depth_sensor_map.DepthObstacleMemory=BoundedDepthMemory
    for directory in (HERE,HERE.parent/'global_repair_v1'):
        harness._PATCH_MODULE_PATHS.extend(f for f in directory.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.sh'))
    def optimize(reference,occupancy,info,*args,**kwargs):
        validate_guard_input(kwargs['reliability'],info)
        return boundary_optimize(reference,occupancy,info,*args,**kwargs)
    harness.optimize=optimize
    loop=harness._control_loop
    def control(ctx,output,session,report,*args,**kwargs):
        report.update(experiment='global_support_bounded_unknown_v2',neural_query_retirement=False,recorded_prefix_used=False,
            generalization_certified=False,global_repair_complete=False,perception_coverage='existing three-camera rig',
            unknown_boundary_policy=dict(policy='support_bounded_unknown_boundary_v2',full_hypotheses_retained=True,measured_geometry_relaxed=False))
        return loop(ctx,output,session,report,*args,**kwargs)
    harness._control_loop=control

def main():
    configure()
    if sys.argv[1:]==['--check-imports']:
        assert depth_sensor_map.DepthObstacleMemory is BoundedDepthMemory
        assert harness.WORKER.is_file()
        print('Global v2 driver/worker imports ready');return
    request=_hugsim_expand(json.loads(Path(sys.argv[1]).read_text()))
    if request['mode']!='record':raise ValueError('This version requires a fresh record; old branch snapshots are incompatible')
    harness.run_record(request)
if __name__=='__main__':main()
