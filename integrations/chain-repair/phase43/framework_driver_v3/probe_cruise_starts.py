
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
import argparse,json,pickle,time,types,numpy as np
from scipy.optimize import minimize as real_minimize
from unittest.mock import patch
from runtime_configuration import load_vehicle_config
from speed_policy import PlanningGuard
from occupancy_protocol import unpack
import empty_controller as c
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args();c.PlanningGuard=PlanningGuard;v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';f=source/('step_%03d'%args.step);report=_hugsim_expand(json.loads((HERE/('behavior_replays/step_%03d_guard.json'%args.step)).read_text()));previous_result=_hugsim_expand(json.loads((HERE/('behavior_replays/step_%03d_result.json'%args.step)).read_text()));names=[r['seed'] for r in previous_result['trials'] if not r['seed'].startswith('feasibility_restored_')];old={r['seed']:r for r in previous_result['trials']};counter=[0]
with (f/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
with np.load(f/'perception.npz') as z:plan=z['trajectory'].copy();occ,contract=unpack(z)
info['occupancy_contract']=contract;report['planning_input_source']='fresh_model_forecast';report['cruise_speed_preference_mps']=report['requested_cruise_speed_mps'];warm=None
if args.step:
 prev=_hugsim_expand(json.loads((source/('step_%03d'%(args.step-1))/'optimization.json').read_text())).get('selected');warm=np.asarray(prev['controls']) if prev else None
performed=[]
def focused(fun,x0,**kwargs):
 name=names[counter[0]%len(names)];counter[0]+=1
 if name in ('lane_change_left','lane_change_right'):
  begin=time.perf_counter();result=real_minimize(fun,x0,**kwargs);performed.append(dict(seed=name,seconds=time.perf_counter()-begin));return result
 row=old[name];return types.SimpleNamespace(x=np.asarray(row['controls']).ravel(),success=bool(row.get('solver_success')),message='Same saved problem candidate rechecked in focused two-start comparison')
start=time.perf_counter()
with patch.object(c,'minimize',side_effect=focused):result=c.optimize(plan,occ,info,warm=warm,reliability=report,vehicle=v,maxiter=100,_forecast_recovery=bool(previous_result.get('forecast_recovery')))
wall=time.perf_counter()-start;out=HERE/'behavior_replays';(out/('step_%03d_cruise_result.json'%args.step)).write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');summary=dict(step=args.step,selected=selected is not None,minimum_constraint=(selected or {}).get('minimum_constraint'),first_action=(selected or {}).get('controls',[None])[0],newly_solved=performed,wall_seconds=wall,scope='Focused comparison: two new cruising starts freshly solved; other same-input candidates rechecked, not a new closed loop.');(out/('step_%03d_cruise_summary.json'%args.step)).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
