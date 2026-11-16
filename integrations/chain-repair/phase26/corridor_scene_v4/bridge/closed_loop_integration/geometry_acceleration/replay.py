"""Full solver replay, archived real observations, same profile/search/constraints."""

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

import argparse,hashlib,json,pickle,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_edited_closed_loop as bridge

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);parser.add_argument('--variant',choices=['baseline','accelerated'],required=True);parser.add_argument('--profile',choices=['phase34','phase35','phase37'],default='phase34');parser.add_argument('--source-run',type=Path);parser.add_argument('--guard-dir',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 source=bridge.CR/({'phase34':'phase34/global_control_v1','phase35':'phase35/event_mpc_v1','phase37':'phase37/uncertainty_closed_v1'}[args.profile])
 if args.profile=='phase37' and (args.guard_dir is None or args.source_run is None):raise ValueError('Phase37 requires paired --source-run and --guard-dir inputs')
 sys.path.insert(0,str(source))
 import launch
 launch.verify()
 if args.profile=='phase37':
  from diagnostic_admission import validate_launch
  validate_launch()
 import run_driver
 run_driver.configure();h=run_driver.h
 import accelerated
 if args.variant=='accelerated':accelerated.install()
 from runtime_configuration import load_vehicle_config
 from occupancy_protocol import unpack
 archived=(args.source_run or bridge.HERE/'runs/20261106T143547Z').resolve();folder=archived/f'uniad_worker/step_{args.step:03d}'
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as f:plan=f['trajectory'].copy();occ,contract=unpack(f)
 info['occupancy_contract']=contract
 rel=_hugsim_expand(json.loads(((args.guard_dir/(folder.name+'_guard.json')) if args.guard_dir else folder/'guard_memory_report.json').read_text()));old=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));warm=None
 if args.step>0:
  prev=_hugsim_expand(json.loads((archived/f'uniad_worker/step_{args.step-1:03d}/optimization.json').read_text())).get('selected')
  if prev is not None:warm=np.asarray(prev['controls'])
 # The live recovery controller is normal in the selected actual frames.
 from motion_policy import RecoveryController
 h._active_recovery_controller=RecoveryController()
 args.output.mkdir(parents=True,exist_ok=False)
 solver_calls=[];raw_minimize=run_driver.optimized_controller.minimize
 def traced(*pos,**kw):
  began=time.perf_counter();sol=raw_minimize(*pos,**kw);row=dict(index=len(solver_calls),seconds=time.perf_counter()-began,nit=int(sol.nit),nfev=int(sol.nfev),success=bool(sol.success));solver_calls.append(row)
  (args.output/'solver_calls.json').write_text(json.dumps(solver_calls,indent=2)+'\n');print(json.dumps(row),flush=True);return sol
 run_driver.optimized_controller.minimize=traced
 t=time.perf_counter();result=h.optimize(plan,occ,info,warm=warm,maxiter=220,reliability=rel,vehicle=load_vehicle_config(),_reuse_verified_warm=False);seconds=time.perf_counter()-t
 (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 selected=result.get('selected');previous=old.get('selected')
 metrics=dict(profile=args.profile,variant=args.variant,step=args.step,wall_seconds=seconds,selected=selected is not None,seed=(selected or {}).get('seed'),candidate_count=len(result.get('trials',[])),reference_source_run=str(archived),new_gpu_executed=False,search_parameters_changed=False)
 if selected is not None and previous is not None:
  metrics.update(maximum_selected_control_difference=float(np.max(np.abs(np.array(selected['controls'])-np.array(previous['controls'])))),maximum_selected_state_difference=float(np.max(np.abs(np.array(selected['states'])-np.array(previous['states'])))),maximum_selected_lidar_difference=float(np.max(np.abs(np.array(selected['lidar'])-np.array(previous['lidar'])))),objective_difference=float(selected['objective']-previous['objective']))
 if args.variant=='accelerated':metrics['geometry_acceleration']=accelerated.audit();accelerated.uninstall()
 (args.output/'summary.json').write_text(json.dumps(metrics,indent=2)+'\n');launch.verify();print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
