from bootstrap import *
import argparse,json,time,hashlib
from cases import load
from evidence_guard import PlanningGuard
from mpc_reuse_v1 import controller
from mpc_reuse_v1.native_guard import PlanningGuard as Original

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('original','repaired'),required=True);parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
 folder,rel,info,plan,occ,vehicle,states,times=load(args.step)
 controller.PlanningGuard=Original if args.variant=='original' else PlanningGuard
 # No shortened iteration budget or constraints: reproduce the actual call.
 started=time.perf_counter();result=controller.optimize(plan,occ,info,warm=None,maxiter=220,reliability=rel,vehicle=vehicle,_reuse_verified_warm=bool(args.step%5));wall=time.perf_counter()-started
 out=HERE/'validation_runs';out.mkdir(exist_ok=True);name=f'step{args.step:03d}_{args.variant}'
 (out/(name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
 summary=dict(step=args.step,variant=args.variant,wall_seconds=wall,selected=result.get('selected') is not None,reason=result.get('reason'),initial_conflict=result.get('initial_conflict'),admission=result.get('guard',{}).get('proposal_admission'),minimum_constraint=(result.get('selected') or {}).get('minimum_constraint'),source_input_sha256=hashlib.sha256((folder/'guard_memory_report.json').read_bytes()).hexdigest(),maxiter=220,control_executed=False,gpu_executed=False)
 (out/(name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
