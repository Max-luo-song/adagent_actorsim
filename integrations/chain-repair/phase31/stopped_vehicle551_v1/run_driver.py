"""Same global contract + bounded unknown + exact optimized MPC, new scene."""

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
import json,hashlib

def configure():
 base.configure()
 setup=h._setup
 def experiment_setup(arm,output):
  result=setup(arm,output)
  if int(result['spec'].source['scene'])!=551:raise ValueError('Expected scene551')
  speed=arm['initial_speed_mps']
  if isinstance(speed,bool) or not 0<float(speed)<=result['vehicle'].max_speed_mps:raise ValueError('Invalid declared initial speed')
  result['recorded_initial_speed_mps']=result['speed'];result['speed']=float(speed)
  result['signature']=hashlib.sha256((result['signature']+'|initial_speed_mps='+str(result['speed'])).encode()).hexdigest()
  return result
 h._setup=experiment_setup
 control=h._control_loop
 def loop(ctx,output,session,report,*args,**kwargs):
  ctx['snapshot_after_index']=None
  try:return control(ctx,output,session,report,*args,**kwargs)
  finally:
   report.update(experiment='stopped_vehicle551_v1',performance_only_change=False,scene_selection_changed=True,initial_speed_override_mps=ctx['setup']['speed'],recorded_initial_speed_mps=ctx['setup']['recorded_initial_speed_mps'],supplemental_sensors_enabled=False,branch_snapshot_disabled=True,inserted_vehicle_static_world_pose=True,termination_policy='197 live source-time controls or original proxy-overlap/error stop; no early success/stall termination')
 h._control_loop=loop
 h._PATCH_MODULE_PATHS.extend(f for f in HERE.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.sh'))

if __name__=='__main__':
 configure()
 request=_hugsim_expand(json.loads(Path(sys.argv[1]).read_text()));base._TIMING_OUTPUT=Path(request['output'])/'mpc_timing.json'
 h.run_record(request)
