
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
import argparse,json,time,hashlib
from cases import load
from evidence_guard import PlanningGuard
from mpc_reuse_v1 import controller
from mpc_reuse_v1.native_guard import PlanningGuard as Original

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('original','repaired'),required=True);parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
 folder,rel,info,plan,occ,vehicle,states,times=load(args.step)
 controller.PlanningGuard=PlanningGuard
 if args.variant=='repaired':rel=_hugsim_expand(json.loads((HERE/f'validation_runs_v2/guard_input_{args.step:03d}.json').read_text()))
 # No shortened iteration budget or constraints: reproduce the actual call.
 started=time.perf_counter();result=controller.optimize(plan,occ,info,warm=None,maxiter=220,reliability=rel,vehicle=vehicle,_reuse_verified_warm=bool(args.step%5));wall=time.perf_counter()-started
 out=HERE/'optimizer_replays';out.mkdir(exist_ok=True);name=f'step{args.step:03d}_{args.variant}'
 (out/(name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
 summary=dict(step=args.step,variant=args.variant,wall_seconds=wall,selected=result.get('selected') is not None,reason=result.get('reason'),initial_conflict=result.get('initial_conflict'),admission=result.get('guard',{}).get('proposal_admission'),minimum_constraint=(result.get('selected') or {}).get('minimum_constraint'),source_input_sha256=hashlib.sha256((folder/'guard_memory_report.json').read_bytes()).hexdigest(),candidate_evidence_fix_enabled_for_both_variants=True,maxiter=220,control_executed=False,gpu_executed=False)
 (out/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
