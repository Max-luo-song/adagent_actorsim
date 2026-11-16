
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
import json,pickle,time,collections,numpy as np
from mpc_schedule import Scheduler
from runtime_configuration import load_vehicle_config
from speed_policy import PlanningGuard
from mpc_reuse_v1.cached_geometry import EpistemicRisk
from occupancy_protocol import unpack
from temporal_context import AgentlessForecast
import empty_controller as controller
controller.PlanningGuard=PlanningGuard
v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop';rows=[];s=Scheduler()
for folder in sorted((source/'uniad_worker').glob('step_*')):
 if int(folder.name.split('_')[-1])>24:continue
 if not (folder/'control_decision.json').exists() or not (folder/'sensor_metadata.pkl').exists():continue
 with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
 with np.load(folder/'perception.npz') as saved:plan=saved['trajectory'].copy();occ,contract=unpack(saved)
 info['occupancy_contract']=contract;old=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));decision=_hugsim_expand(json.loads((folder/'control_decision.json').read_text()));report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report['planning_input_source']=old['planning_input_source']
 if old.get('selected') is not None:
  offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];braking=[np.r_[-offset,0.,info['ego_velo'],info['ego_steer']]];states=np.asarray(old['selected']['states']);times=np.arange(len(states))*v.dt
  for k in range(len(states)-1):
   acc=0. if k*v.dt<report.get('config',{}).get('confirmation_seconds',.2)-1e-9 else -min(v.emergency_braking_mps2,braking[-1][3]/v.dt);braking.append(controller.integrate_state(braking[-1],[acc,0.],v.dt,v.wheelbase_m))
  guard=PlanningGuard(report,v,np.asarray(braking),times,current_occupancy=np.asarray(occ).reshape(5,200,200)[0] if occ is not None else None);old['selected']['soft_risk_cost']=float(guard.soft_cost(states,times)+EpistemicRisk(report,info,v).cost(states))
 req=s.request(plan,info,v,old['planning_input_source'],decision['state_before']);row=dict(step=int(folder.name.split('_')[-1]),requested=req['requested'],trigger=req['trigger'])
 if req['requested']:
  if occ is None and old['planning_input_source']=='fresh_agentless_foreground':occ=AgentlessForecast();report['agentless_foreground_contract']=old['perception_context_audit']
  start=time.perf_counter();result=controller.optimize(plan,occ,info,vehicle=v,reliability=report,_continuation=req,_check_continuation_only=True);row.update(check_wall_seconds=time.perf_counter()-start,accepted=result.get('selected') is not None,audit=result['guard']['scheduling'])
  out=HERE/'schedule_replays';out.mkdir(exist_ok=True);(out/(folder.name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
  if not row['accepted']:s.record(old,plan,info)
 else:s.record(old,plan,info)
 rows.append(row);print(json.dumps(row),flush=True)
(HERE/'schedule_replay_results.json').write_text(json.dumps(dict(scope='Counterfactual schedule checks on archived v2 observations. Later ego motion follows archived actions, not the new continued plan; not a closed-loop outcome.',rows=rows,accepted=sum(x.get('accepted',False) for x in rows),new_gpu_executed=False),indent=2)+'\n')
