"""Actual saved no-query transition; CPU counterfactual, not a closed loop."""

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
import json,pickle,time,numpy as np
from run_driver import configure
from occupancy_protocol import unpack,encode
from runtime_configuration import load_vehicle_config
from motion_policy import RecoveryController
configure();source=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop/uniad_worker';previous=None;chosen=None
for folder in sorted(source.glob('step_[0-9][0-9][0-9]')):
 audit=folder/'occupancy_head_audit.json'
 if not audit.exists():continue
 count=_hugsim_expand(json.loads(audit.read_text()))['motion_query_count']
 if count==0 and previous is not None and previous[1]>0:
  if chosen is None:chosen=(previous[0],folder)
  with np.load(previous[0]/'perception.npz') as data:
   if 'occupancy' in data and not np.any(data['occupancy']):chosen=(previous[0],folder);break
 previous=(folder,count)
if chosen is None:raise RuntimeError('No observed valid-to-no-query transition found')
def load(folder):
 with (folder/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
 del obs
 with np.load(folder/'perception.npz') as data:
  plan=data['trajectory'].copy()
  if 'occupancy_contract_json' in data:occupancy,contract=unpack(data)
  else:occupancy=data['occupancy'].copy() if 'occupancy' in data else None;contract=encode(occupancy,0 if occupancy is None else None)[1]
 info['occupancy_contract']=contract
 from forecast_memory import frame_health
 audit_path=folder/'occupancy_head_audit.json';audit=_hugsim_expand(json.loads(audit_path.read_text())) if audit_path.exists() else {}
 with np.load(folder/'perception.npz') as data:info['perception_health']=frame_health(info['timestamp'],audit,plan,[data['candidate_'+key] for key in ['boxes','scores','labels']],True)
 report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));return plan,occupancy,info,report
before,current=chosen;old_plan,masks,old_info,old_report=load(before);plan,occupancy,info,report=load(current)
if masks is None or occupancy is not None or info['occupancy_contract']['reason']!='no_motion_queries':raise ValueError('Wrong dropout fixture')
h._forecast_memory.resolve(masks,old_info,old_report);controller=RecoveryController();h._active_recovery_controller=controller;v=load_vehicle_config();start=time.monotonic();result=h.optimize(plan,None,info,reliability=report,vehicle=v,maxiter=40);wall=time.monotonic()-start;action,decision=controller.update(result,info['ego_velo'],info['timestamp'],v.dt,v.emergency_braking_mps2)
assert result['planning_input_source'] in ['retained_model_forecast','fresh_agentless_foreground']
assert result['occupancy_input_audit']['status']=='unavailable'
if result['planning_input_source']=='retained_model_forecast':assert action['acc']<=1e-5
assert v.min_acc_mps2-1e-8<=action['acc']<=v.max_acc_mps2+1e-8
summary={'source_previous_step':before.name,'source_missing_step':current.name,'wall_seconds':wall,'selected':result['selected'] is not None,'reason':result.get('reason'),'retained_age_s':(result.get('forecast_memory_audit') or {}).get('age_s'),'raw_current_status_preserved':result['occupancy_input_audit']['status'],'planning_input_source':result['planning_input_source'],'minimum_constraint':(result.get('selected') or {}).get('minimum_constraint'),'action':action,'decision':decision,'current_raster_fabricated':False,'counterfactual_initial_controller_state':'normal','new_gpu_executed':False,'new_closed_loop_executed':False}
(HERE/'dropout_replay_result.json').write_text(json.dumps(result,indent=2)+'\n');(HERE/'dropout_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
