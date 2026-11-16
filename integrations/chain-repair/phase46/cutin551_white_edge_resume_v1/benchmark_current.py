"""Current cut-in inputs; distinct numerical and search-policy experiments."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,copy,inspect,json,pickle,time
import numpy as np
from paths import *
from run_cutin import configure
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);parser.add_argument('--variant',choices=['baseline','ranked','joint','combined'],required=True);parser.add_argument('--probe',action='store_true');args=parser.parse_args();driver=configure();opt=driver.optimized_controller
with (HERE/'validation/inputs'/('step_%03d.pkl'%args.step)).open('rb') as f:data=pickle.load(f)
policy=_hugsim_expand(json.loads((DRIVER/'runtime_policy.json').read_text()));policy.update(ranked_expansion=args.variant in ('ranked','combined'),joint_difference=args.variant in ('joint','combined'));kwargs=data['kwargs'];kwargs['_runtime_policy']=policy
out=HERE/'validation/current';out.mkdir(parents=True,exist_ok=True)
if args.probe:
 from joint_difference import JointDifference
 from scipy.optimize._numdiff import approx_derivative
 from scipy.optimize._slsqp_py import _epsilon
 captured={}
 class Captured(Exception):pass
 def capture(fun,x0,**kwargs):captured.update(objective=fun,constraints=kwargs['constraints'][0]['fun'],bounds=kwargs['bounds']);raise Captured()
 original=opt.minimize;opt.minimize=capture
 try:opt.optimize(data['reference'],data['occupancy'],data['info'],**kwargs)
 except Captured:pass
 finally:opt.minimize=original
 x=np.asarray(data['source_result']['selected']['controls']).ravel();bounds=np.asarray(captured['bounds']).T;engine=JointDifference(**captured);start=time.perf_counter();a=approx_derivative(captured['objective'],x,method='2-point',abs_step=_epsilon,bounds=bounds);b=approx_derivative(captured['constraints'],x,method='2-point',abs_step=_epsilon,bounds=bounds);old=time.perf_counter()-start;start=time.perf_counter();aa=engine.gradient(x);bb=engine.jacobian(x);new=time.perf_counter()-start
 assert a.tobytes()==aa.reshape(1,-1).tobytes();assert b.tobytes()==bb.tobytes();result=dict(step=args.step,exact_gradient_equality=True,scalars=a.size+b.size,baseline_seconds=old,joint_seconds=new,metrics=engine.metrics,timing_not_comparable_shared_warm_cache=True)
 (out/('gradient_%03d.json'%args.step)).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
else:
 start=time.perf_counter();cpu=time.process_time();result=opt.optimize(data['reference'],data['occupancy'],data['info'],**kwargs);wall=time.perf_counter()-start;cpu=time.process_time()-cpu;selected=result.get('selected');source=data['source_result'].get('selected');summary=dict(step=args.step,variant=args.variant,wall_seconds=wall,process_cpu_seconds=cpu,objective=(selected or {}).get('objective'),minimum_constraint=(selected or {}).get('minimum_constraint'),seed=(selected or {}).get('seed'),trials=len(result.get('trials',[])),source_plan_bitwise_equal=selected is not None and source is not None and np.array_equal(selected['controls'],source['controls']),source_objective=(source or {}).get('objective'),source_selected_feasible=bool(source and source['feasible']),forecast_recovery=result.get('forecast_recovery'),gpu_executed=False)
 (out/('%s_%03d_result.json'%(args.variant,args.step))).write_text(json.dumps(result,indent=2)+'\n');(out/('%s_%03d_summary.json'%(args.variant,args.step))).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
