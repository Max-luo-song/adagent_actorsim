
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
from drivable_area import DrivableArea
from road_soft_cost import SoftRoadPenalty
configure();parser=argparse.ArgumentParser();parser.add_argument('--step',required=True,type=int);args=parser.parse_args();source=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop';folder=source/f'uniad_worker/step_{args.step:03d}'
with (folder/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
del obs
with np.load(folder/'perception.npz') as saved:plan=saved['trajectory'].copy();occ=saved['occupancy'].copy()
report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));v=load_vehicle_config();old=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));warm=None
if args.step>0:
 previous=_hugsim_expand(json.loads((source/f'uniad_worker/step_{args.step-1:03d}/optimization.json').read_text())).get('selected')
 if previous:warm=np.asarray(previous['controls'])
started=time.perf_counter();result=h.optimize(plan,occ,info,warm=warm,maxiter=220,reliability=report,vehicle=v,_reuse_verified_warm=True);wall=time.perf_counter()-started
out=HERE/'current_replays';out.mkdir(exist_ok=True);(out/f'step_{args.step:03d}_result.json').write_text(json.dumps(result,indent=2)+chr(10))
road=DrivableArea(report['predicted_drivable_area'],v);soft=SoftRoadPenalty(road,v)
def metrics(result):
 selected=result.get('selected')
 if selected is None:return dict(selected=False,reason=result.get('reason'))
 states=np.asarray(selected['states']);gap=road.clearance(states);tracking=np.asarray(selected['lidar'])[5::5][:6];error=np.linalg.norm(tracking-plan,axis=1)
 return dict(selected=True,seed=selected['seed'],minimum_constraint=selected['minimum_constraint'],unweighted_road_penalty=float(np.sum(np.minimum(gap,0.)**2)/5),road_distance_penalty=soft.distance_cost(states,5),combined_unweighted_road_cost=soft.cost(states,5),negative_road_samples=int((gap<0).sum()),endpoint_road_gap_m=float(gap[-1]),current_frame_tracking_error_mean_m=float(error.mean()),current_frame_tracking_error_max_m=float(error.max()),first_control=selected['controls'][0])
summary=dict(step=args.step,before=metrics(old),after=metrics(result),wall_seconds=wall,road_soft_cost_weight=10.,reference_source=result.get('guard',{}).get('navigation_reference',{}).get('source'),same_saved_observation=True,new_gpu_executed=False,new_closed_loop_executed=False)
if result.get('selected') is not None:
 selected=result['selected'];assert selected['feasible'] and selected['minimum_constraint']>=-1e-5
 audit=result['guard']['navigation_reference'];np.testing.assert_array_equal(audit['original_neural_reference'],plan);np.testing.assert_array_equal(np.asarray(audit['tracking_reference'])[:6],plan)
(out/f'step_{args.step:03d}_summary.json').write_text(json.dumps(summary,indent=2)+chr(10));print(json.dumps(summary,indent=2),flush=True)
if not result.get('selected'):raise SystemExit(2)
