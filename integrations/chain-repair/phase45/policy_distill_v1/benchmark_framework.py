"""Own paired CPU integration evidence on real Framework observations."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,json,pickle,sys,time
import numpy as np
from paths import HERE,DRIVER,PHASE26
from run_cutin import configure
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=11);parser.add_argument('--variant',choices=['baseline','candidate'],default='candidate');parser.add_argument('--equivalence-only',action='store_true');args=parser.parse_args();driver=configure();opt=driver.optimized_controller
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
source=PHASE26/'bridge/closed_loop_integration/runs/20261107T064934Z/uniad_worker';folder=source/('step_%03d'%args.step)
with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
with np.load(folder/'perception.npz') as z:plan=z['trajectory'].copy();occ,contract=unpack(z)
info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((PHASE26/'motion_envelope_v2/envelope_evidence'/('step_%03d_guard.json'%args.step)).read_text()));previous=_hugsim_expand(json.loads((source/('step_%03d'%(args.step-1))/'optimization.json').read_text())).get('selected');warm=None if previous is None else np.asarray(previous['controls']);base=_hugsim_expand(json.loads((DRIVER/'runtime_policy.json').read_text()));out=HERE/'validation/benchmarks';out.mkdir(parents=True,exist_ok=True)
def policy(enabled):return dict(base,compiled_polygon=True,rollout_prefix_reuse=enabled,batched_occupancy=enabled,bulk_pose_cache=enabled)
if args.equivalence_only:
 functions=[]
 class Captured(Exception):pass
 for enabled in (False,True):
  saved={}
  def capture(fun,x0,**kwargs):saved.update(objective=fun,constraints=kwargs['constraints'][0]['fun']);raise Captured()
  original=opt.minimize;opt.minimize=capture
  try:opt.optimize(plan,occ,info,warm=warm,maxiter=220,reliability=report,vehicle=load_vehicle_config(),_warm_elapsed_s=.1,_runtime_policy=policy(enabled))
  except Captured:pass
  finally:opt.minimize=original
  functions.append(saved)
 x=np.asarray(_hugsim_expand(json.loads((PHASE26/'motion_search_v3/search_benchmarks/candidate'/('step_%03d_result.json'%args.step)).read_text()))['selected']['controls']).ravel();count=0
 for j in range(13):
  q=x.copy()
  if j<12:q[j]+=np.sqrt(np.finfo(float).eps)
  for name in ('objective','constraints'):
   a=np.asarray(functions[0][name](q));b=np.asarray(functions[1][name](q));assert a.tobytes()==b.tobytes(),name;count+=a.size
 result=dict(passed=True,step=args.step,scalar_values_compared=count,bitwise_equal=True,new_gpu_executed=False);(out/('step_%03d_equivalence.json'%args.step)).write_text(json.dumps(result,indent=2)+'\n');print(result)
else:
 start=time.perf_counter();cpu=time.process_time();result=opt.optimize(plan,occ,info,warm=warm,maxiter=220,reliability=report,vehicle=load_vehicle_config(),_warm_elapsed_s=.1,_runtime_policy=policy(args.variant=='candidate'));wall=time.perf_counter()-start;cpu=time.process_time()-cpu;selected=result.get('selected');summary=dict(step=args.step,variant=args.variant,wall_seconds=wall,process_cpu_seconds=cpu,selected=selected is not None,objective=(selected or {}).get('objective'),minimum_constraint=(selected or {}).get('minimum_constraint'),forecast_recovery=result.get('forecast_recovery'),trials=len(result.get('trials',[])),new_gpu_executed=False)
 (out/('%s_step_%03d_result.json'%(args.variant,args.step))).write_text(json.dumps(result,indent=2)+'\n');(out/('%s_step_%03d_summary.json'%(args.variant,args.step))).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
