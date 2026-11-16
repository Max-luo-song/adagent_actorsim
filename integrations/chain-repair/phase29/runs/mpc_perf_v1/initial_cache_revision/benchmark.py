"""Isolated same-input CPU comparison; no GPU/runtime source modification."""

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

import argparse,datetime,hashlib,json,pickle,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(HERE.parent)]
CASES={
 'corrected36':(CR/'phase29/runs/global_evidence_closed552_v1',36),
 'slow2':(CR/'phase29/runs/global_evidence_closed552_v1',2),
 'normal15':(CR/'phase29/runs/global_evidence_closed552_v1',15),
 'scene172':(CR/'phase15/runs/20261025T093932_full/fast_gate/172_unedited/baseline',0),
 'scene621':(CR/'phase15/runs/20261025T093932_full/fast_gate/621_unedited/baseline',0),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True);p.add_argument('--variant',choices=('baseline','perf','cache_only','batch_only'),required=True);a=p.parse_args()
 if a.variant=='baseline':
  from global_repair_v2 import controller
 else:
  from mpc_perf_v1 import controller
  if a.variant=='batch_only':controller.Memo.wrap=lambda self,fn:fn
  if a.variant=='cache_only':
   from mpc_perf_v1.fast_guard import PlanningGuard,OriginalPlanningGuard
   PlanningGuard.clearance_independent=lambda self,states,time=0.:np.array([OriginalPlanningGuard.clearance(self,s[None],[time])[0] for s in states])
 run,index=CASES[a.case];folder=run/f'uniad_worker/step_{index:03d}'
 path=folder/'sensor_metadata.pkl'
 if path.exists():
  with path.open('rb') as f:info=pickle.load(f)
 else:
  with (folder/'observation.pkl').open('rb') as f:_,info=pickle.load(f)
 relpath=CR/'phase29/runs/bounded_unknown_validation_v2/guard_input_step036.json' if a.case=='corrected36' else folder/'guard_memory_report.json'
 rel=_hugsim_expand(json.loads(relpath.read_text()))
 with np.load(folder/'perception.npz') as data:plan=data['trajectory'].copy();occupancy=data['occupancy'].copy()
 warm=None
 if index:
  prev=_hugsim_expand(json.loads((run/f'uniad_worker/step_{index-1:03d}/optimization.json').read_text()));warm=(prev.get('selected') or {}).get('controls')
 out=CR/'phase29/runs/mpc_perf_v1';out.mkdir(exist_ok=True);stem=a.case+'_'+a.variant+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 print('START',a.case,a.variant,flush=True);start=time.perf_counter()
 result=controller.optimize(plan,occupancy,info,warm=warm,maxiter=220,reliability=rel,_reuse_verified_warm=bool(index%5))
 wall=time.perf_counter()-start
 summary=dict(case=a.case,variant=a.variant,wall_seconds=wall,maxiter=220,selected=result.get('selected') is not None,
   selected_seed=(result.get('selected') or {}).get('seed'),forecast_recovery=bool(result.get('forecast_recovery')),
   control_interval_s=(result.get('selected') or {}).get('control_interval_s'),
   source=str(run),step=index,guard_input_sha256=hashlib.sha256(relpath.read_bytes()).hexdigest(),
   performance_cache_layers=result.get('performance_cache_layers',[]),result_file=str(out/(stem+'_result.json')),
   scope='Fixed-input CPU optimize comparison; no rendering, NN inference or new control execution')
 (out/(stem+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');(out/(stem+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
