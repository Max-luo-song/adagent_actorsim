
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
import argparse,json,pickle,time,numpy as np
from run_driver import configure
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
import empty_controller
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=1);parser.add_argument('--variant',choices=['baseline','prefix','occupancy','cache','candidate','full'],default='candidate');parser.add_argument('--source',choices=['phase39','phase37'],default='phase39');parser.add_argument('--output-folder',default='runtime_benchmarks');args=parser.parse_args();configure();source=CR/'phase39/runs/mpc_ten551_20261107T112411Z_83f645be/closed_loop/uniad_worker';source=source if args.source=='phase39' else CR/'phase37/runs/uncertainty_closed551_20261107T044006Z_bc47a553/closed_loop/uniad_worker';folder=source/('step_%03d'%args.step)
with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
with np.load(folder/'perception.npz') as z:occ,contract=unpack(z);reference=z['trajectory'].copy()
info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report['cruise_speed_preference_mps']=8.;warm=None
if args.step:
 previous=_hugsim_expand(json.loads((source/('step_%03d'%(args.step-1))/'optimization.json').read_text())).get('selected');warm=np.asarray(previous['controls']) if previous else None
recorded=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));iteration_limit=max(x.get('iteration_limit',220) for x in recorded['trials'])
policy=dict(compiled_road=True,scaled_slsqp=False,consensus_search=args.variant!='full',adaptive_consensus=True,compiled_polygon=True,rollout_prefix_reuse=args.variant in ('prefix','candidate','full'),batched_occupancy=args.variant in ('occupancy','candidate','full'),bulk_pose_cache=args.variant in ('cache','candidate','full'));started=time.perf_counter();cpu_started=time.process_time();from polygon_backend import use_backend
with use_backend(policy['compiled_polygon']):result=empty_controller.optimize(reference,occ,info,reliability=report,vehicle=load_vehicle_config(),warm=warm,maxiter=iteration_limit,_warm_elapsed_s=.1 if warm is not None else None,_runtime_policy=policy)
wall=time.perf_counter()-started;out=HERE/args.output_folder;out.mkdir(exist_ok=True);stem='step_%03d_%s'%(args.step,args.variant);(out/(stem+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');summary=dict(step=args.step,variant=args.variant,wall_seconds=wall,process_cpu_seconds=time.process_time()-cpu_started,selected=selected is not None,objective=(selected or {}).get('objective'),first_action=(selected or {}).get('controls',[None])[0],minimum_constraint=(selected or {}).get('minimum_constraint'),forecast_recovery=result.get('forecast_recovery'),iterations=sum(x.get('solver_iterations',0) or 0 for x in result.get('trials',[])),evaluations=sum(x.get('objective_evaluations',0) or 0 for x in result.get('trials',[])),trials=len(result.get('trials',[])),scope='Full CPU solve on one immutable current-run observation; no new closed loop.');(out/(stem+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
