
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import json,numpy as np,tempfile
from ten_step_driver import configure

def main():
 configure();arm=h._load_arm_request(HERE/'arm.json')
 with tempfile.TemporaryDirectory(prefix='scene551-gate-') as folder:
  s=h._setup(arm,Path(folder));state=h._fresh_driver_state(s,s['spec'])
  assert s['spec'].source['scene']==551 and s['spec'].frames==199
  assert s['start']==0 and s['steps']==10 and s['steps']<s['spec'].frames
  assert s['speed']==8. and state['motion'].speed==8.
  assert s['recorded_initial_speed_mps']>15.
  assert np.allclose(s['relative'][0],np.eye(4))
  assert s['vehicle'].min_acc_mps2==-2. and s['vehicle'].clearance_margin_m==.2
  from depth_sensor_map import DepthObstacleMemory
  from depth_consistency import DepthConsistencyMemory as BoundedDepthMemory
  assert DepthObstacleMemory is BoundedDepthMemory
  assert not any(k.startswith('surround_depth_v1') for k in sys.modules)
  assert base.boundary_optimize.__module__=='empty_controller'
  from speed_policy import PlanningGuard
  assert base.boundary_optimize.__globals__['PlanningGuard'] is PlanningGuard
  assert h.WORKER==HERE/'worker.py'
  assert s['insert']['object_id']=='stopped_vehicle551'
  assert not h.reference_check(s['initial_native'],s['reference'],s['vehicle'])['overlap']
  assert all(Path(s['spec'].source[k]).exists() for k in ('checkpoint','inputs','source_config'))
  result=dict(passed=True,scene=551,declared_initial_speed_mps=8.,actual_motion_initial_speed_mps=state['motion'].speed,recorded_initial_speed_mps=s['recorded_initial_speed_mps'],worker=str(h.WORKER),mpc_backend=base.boundary_optimize.__module__,surround_sensor_loaded=False,controls_executed=False,gpu_executed=False,runtime_performance_policy=_hugsim_expand(json.loads((HERE/'runtime_policy.json').read_text())))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
