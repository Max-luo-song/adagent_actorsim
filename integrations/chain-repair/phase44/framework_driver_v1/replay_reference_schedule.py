
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
import copy,json,pickle,time,numpy as np
from mpc_schedule import Scheduler,POLICY
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
from run_driver import configure
import empty_controller
configure();v=load_vehicle_config();root=CR/'phase37/runs/uncertainty_closed551_20261107T044006Z_bc47a553/closed_loop';source=root/'uniad_worker';rows=[];schedulers={}
for kind in ['timed','spatial_arc']:
 policy=dict(POLICY,reference_metric=kind);schedulers[kind]=Scheduler(policy)
for folder in sorted(source.glob('step_*')):
 if not all((folder/name).exists() for name in ('optimization.json','sensor_metadata.pkl','perception.npz','guard_memory_report.json')):continue
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:occ,contract=unpack(z);reference=z['trajectory'].copy()
 info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report['cruise_speed_preference_mps']=8.;actual=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));row=dict(step=int(folder.name.split('_')[-1]),actual_mode=actual.get('guard',{}).get('scheduling',{}).get('mode'))
 for kind,scheduler in schedulers.items():
  request=scheduler.request(reference,info,v,actual.get('planning_input_source','fresh_model_forecast'));record=dict(requested=request['requested'],trigger=request['trigger'],reference_error_m=request.get('reference_error_m'))
  if request['requested']:
   started=time.perf_counter();result=empty_controller.optimize(reference,occ,info,reliability=report,vehicle=v,_continuation=request,_check_continuation_only=True,_runtime_policy={'compiled_road':True,'scaled_slsqp':False});selected=result.get('selected');record.update(check_seconds=time.perf_counter()-started,accepted=selected is not None,check_trigger=result['guard']['scheduling']['trigger'],minimum_constraint=(selected or {}).get('minimum_constraint'),current_objective=(selected or {}).get('objective'))
  row[kind]=record;scheduler.record(actual,reference,info)
 rows.append(row)
summary=dict(frames=len(rows),rows=rows,scope='Counterfactual one-frame continuation checks using actual saved anchors/states; no simulated optimized trajectory sequence.',new_gpu_executed=False)
(HERE/'reference_schedule_replay.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
