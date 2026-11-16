
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
import empty_controller as controller
from evidence_guard import PlanningGuard
from constraint_audit import initial_components
from empty_native_temporal import TemporalOccupancy
from planning_guard import RecoveryController
SOURCE=CR/'phase31/runs/depth_consistency551_20261105T051937Z_fc796426/closed_loop'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args();f=SOURCE/f'uniad_worker/step_{args.step:03d}'
 with (f/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
 del obs
 with np.load(f/'perception.npz') as z:plan=z['trajectory'].copy();occ=z['occupancy'].copy()
 assert not np.any(occ)
 rel=_hugsim_expand(json.loads((f/'guard_memory_report.json').read_text()));v=controller.load_vehicle_config();controller.PlanningGuard=PlanningGuard
 started=time.perf_counter();result=controller.optimize(plan,occ,info,warm=None,maxiter=220,reliability=rel,vehicle=v,_reuse_verified_warm=bool(args.step%5));wall=time.perf_counter()-started
 result['initial_geometry_components']=initial_components(occ,rel,info,v)
 selected=result.get('selected');assert selected is not None,'No feasible result: '+str(result.get('reason'))
 states=np.asarray(selected['states']);times=np.arange(len(states))*v.dt
 assert selected['feasible'] and selected['minimum_constraint']>=-1e-5
 recovery=RecoveryController();action,decision=recovery.update(result,float(info['ego_velo']),float(info['timestamp']),v.dt,v.emergency_braking_mps2)
 summary=dict(step=args.step,passed=True,valid_empty=True,selected=True,minimum_constraint=selected['minimum_constraint'],wall_seconds=wall,initial_geometry_components=result['initial_geometry_components'],decision=decision,action=action,neural_threshold_changed=False,gpu_executed=False,closed_loop_executed=False,maxiter=220)
 out=HERE/'replays';out.mkdir(exist_ok=True);(out/f'step_{args.step:03d}_result.json').write_text(json.dumps(result,indent=2)+'\n');(out/f'step_{args.step:03d}_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
