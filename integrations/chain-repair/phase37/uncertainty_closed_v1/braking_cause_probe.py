
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
import json,pickle,copy,numpy as np
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
from speed_policy import PlanningGuard
from context_factory import make_temporal
from empty_controller import rollout,bounded_seed,sweep_allowance
from drivable_area import DrivableArea
from road_soft_cost import SoftRoadPenalty
from native_objective import NativeAlignedObjective
from mpc_reuse_v1.cached_geometry import EpistemicRisk
from spatial_reference import spatial_error,POLICY
v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker/step_000'
with (source/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
with np.load(source/'perception.npz') as z:occ,_=unpack(z);plan=z['trajectory'].copy()
report=_hugsim_expand(json.loads((HERE/'behavior_replays/step_000_guard.json').read_text()));offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];rows=[]
for mode in ['all','no_depth_hard_DIAGNOSTIC','no_depth_or_unknown_DIAGNOSTIC']:
 r=copy.deepcopy(report)
 if mode!='all':r['observed_depth_obstacles_xy']=[]
 if mode=='no_depth_or_unknown_DIAGNOSTIC':r['unknown_visibility_obstacles_xy']=[];r['unknown_visibility_geometry']['epistemic_only_points_xy']=[]
 temporal=make_temporal(np.asarray(occ).reshape(1,5,200,200),info,r,v);brake,_=rollout(bounded_seed(np.tile([-2.,0.],(6,1)),info['ego_velo'],info['ego_steer'],1.,v),info['ego_velo'],info['ego_steer'],offset,control_interval_s=1.);times=np.arange(len(brake))*v.dt;guard=PlanningGuard(r,v,brake,times,current_occupancy=np.asarray(occ).reshape(5,200,200)[0]);road=SoftRoadPenalty(DrivableArea(r['predicted_drivable_area'],v),v)
 clear=lambda states:np.minimum(temporal.clearance(states,np.arange(len(states))*v.dt),guard.clearance(states,np.arange(len(states))*v.dt))-sweep_allowance(states,v)
 objective=NativeAlignedObjective(plan,v,road,temporal,guard,clear);objective.cruise_preference=info['ego_velo'];objective.epistemic=EpistemicRisk(r,info,v)
 ref=np.vstack([plan,plan[-1]+(np.arange(3.5,6.01,.5)[:,None]-3.)*(plan[-1]-plan[-2])/.5])
 for acc in [-2.,-1.,-.5,0.]:
  controls=bounded_seed(np.tile([acc,0.],(6,1)),info['ego_velo'],info['ego_steer'],1.,v);states,lidar=rollout(controls,info['ego_velo'],info['ego_steer'],offset,control_interval_s=1.);metrics=objective.components(states,lidar,controls)
  rows.append(dict(mode=mode,acceleration=acc,minimum_gap=float(clear(states).min()),cost=objective.cost(states,lidar,controls,ref),speed_cost=float(np.mean((states[1:,3]-info['ego_velo'])**2)),spatial_cost=spatial_error(lidar[5::5],plan,objective.stop),metrics=metrics))
(HERE/'braking_cause_probe.json').write_text(json.dumps(dict(scope='Offline causal ablation only; obstacle removals are never enabled online.',rows=rows),indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='metrics'} for x in rows],indent=2))
