
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
import json,pickle,numpy as np
from speed_policy import PlanningGuard
from context_factory import make_temporal
from depth_ambiguity import ObservationViability
from runtime_configuration import load_vehicle_config
from empty_controller import integrate_state,sweep_allowance
from occupancy_protocol import unpack
v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';out=HERE/'depth_behavior_replays/551';rows=[]
for report_path in sorted(out.glob('step_*_guard.json')):
 report=_hugsim_expand(json.loads(report_path.read_text()));name=report_path.name.replace('_guard.json','');folder=source/name
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:occ,_=unpack(z)
 initial=np.r_[-np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3],0.,info['ego_velo'],info['ego_steer']];braking=[initial.copy()]
 for k in range(60):
  state=braking[-1];acc=0. if k*.1<report.get('config',{}).get('confirmation_seconds',.2)-1e-9 else -min(2.,state[3]/.1);braking.append(integrate_state(state,[acc,0.],.1,v.wheelbase_m))
 guard=PlanningGuard(report,v,np.asarray(braking),np.arange(61)*.1,current_occupancy=np.asarray(occ).reshape(5,200,200)[0]);temporal=make_temporal(np.asarray(occ).reshape(1,5,200,200),info,report,v);budget=ObservationViability(report,info,v,motion_guard=guard)
 budget.known_clear=lambda states:np.minimum(temporal.clearance(states,np.arange(len(states))*.1),guard.full_clearance(states,np.arange(len(states))*.1))-sweep_allowance(states,v)
 probes=[]
 for acc in [0.,-min(2.,initial[3]/.1)]:
  for rate in [0.,-np.deg2rad(v.max_steer_rate_degrees),np.deg2rad(v.max_steer_rate_degrees)]:
   if abs(initial[4]+rate*.1)>np.deg2rad(v.max_steer_degrees):continue
   gap=budget.gap(initial,np.array([acc,rate]));probes.append(dict(acceleration=acc,steering_rate=rate,gap_m=(gap if np.isfinite(gap) else None),viable=bool(gap>=-1e-5)))
 row=dict(step=int(name.split('_')[-1]),coast_first_action_viable=any(x['viable'] for x in probes if x['acceleration']==0.),some_braking_first_action_viable=any(x['viable'] for x in probes if x['acceleration']<0.),probes=probes);rows.append(row)
summary=dict(frames=len(rows),coast_viable_frames=sum(x['coast_first_action_viable'] for x in rows),no_coast_witness_steps=[x['step'] for x in rows if not x['coast_first_action_viable']],rows=rows,scope='Grid of first actions on recorded old states, with complete robust-envelope braking witnesses; not a new policy, optimized action sequence or new closed loop.',new_gpu_executed=False)
(out/'action_viability_sequence.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
