
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
import json,pickle,time,types,numpy as np
from unittest.mock import patch
from runtime_configuration import load_vehicle_config
from speed_policy import PlanningGuard
from occupancy_protocol import unpack
import empty_controller as controller
controller.PlanningGuard=PlanningGuard
source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';folder=source/'step_009';old=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
with np.load(folder/'perception.npz') as saved:plan=saved['trajectory'].copy();occ,contract=unpack(saved)
info['occupancy_contract']=contract;report['planning_input_source']='fresh_model_forecast';report['cruise_speed_preference_mps']=8.;v=load_vehicle_config();previous=_hugsim_expand(json.loads((source/'step_008/optimization.json').read_text()))['selected'];iterator=iter(old['trials'])
def archived_failed_candidate(fun,x0,**kwargs):
 row=next(iterator);return types.SimpleNamespace(x=np.asarray(row['controls']).ravel(),success=False,message='Archived infeasible solution replayed as phase-I start')
start=time.perf_counter()
with patch.object(controller,'minimize',side_effect=archived_failed_candidate):result=controller.optimize(plan,occ,info,warm=np.asarray(previous['controls']),maxiter=100,reliability=report,vehicle=v,_forecast_recovery=True)
wall=time.perf_counter()-start;(HERE/'restoration_replay_result.json').write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');summary=dict(scope='Phase-I diagnostic initialized from archived failed candidates under the same unrelaxed step9 constraints; not a fresh full search or new closed loop.',selected=selected is not None,minimum_constraint=(selected or {}).get('minimum_constraint'),first_action=(selected or {}).get('controls',[None])[0],wall_seconds=wall,restoration=result['guard'].get('feasibility_restoration'),new_gpu_executed=False)
(HERE/'restoration_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
