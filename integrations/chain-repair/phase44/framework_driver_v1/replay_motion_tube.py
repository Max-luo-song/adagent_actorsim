
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
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
from context_factory import make_temporal
from empty_controller import integrate_state,sweep_allowance
v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker/step_009';report=_hugsim_expand(json.loads((HERE/'memory_replays/step_009_guard.json').read_text()))
with (source/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
with np.load(source/'perception.npz') as z:occ,_=unpack(z)
off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];braking=[np.r_[-off,0.,info['ego_velo'],info['ego_steer']]]
for k in range(60):
 acc=0. if k*.1<report.get('config',{}).get('confirmation_seconds',.2)-1e-9 else -min(v.emergency_braking_mps2,braking[-1][3]/.1);braking.append(integrate_state(braking[-1],[acc,0.],.1,v.wheelbase_m))
times=np.arange(61)*.1;guard=PlanningGuard(report,v,np.asarray(braking),times,current_occupancy=np.asarray(occ).reshape(5,200,200)[0]);temporal=make_temporal(np.asarray(occ).reshape(1,5,200,200),info,report,v);old=_hugsim_expand(json.loads((source/'optimization.json').read_text()));rows=[]
for row in old['trials']:
 states=np.asarray(row['states']);t=np.arange(len(states))*.1;sweep=sweep_allowance(states,v);gap=float((np.minimum(guard.clearance(states,t),temporal.clearance(states,t))-sweep).min());physical=min(float(states[:,3].min()),float((v.max_speed_mps-states[:,3]).min()),float((np.deg2rad(v.max_steer_degrees)-np.abs(states[:,4])).min()));rows.append(dict(seed=row['seed'],old_minimum=row['minimum_constraint'],new_geometry_gap=gap,physical_minimum=physical,feasible=gap>=-1e-5 and physical>=-1e-5))
summary=dict(qualified_motion_targets=len(guard.local_motion),certificates=list(guard.local_motion.values()),rechecked_saved_candidates=rows,scope='Current-code motion geometry on same archived perception; no target deletion or margin reduction, not a closed loop.',new_gpu_executed=False);(HERE/'motion_tube_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
