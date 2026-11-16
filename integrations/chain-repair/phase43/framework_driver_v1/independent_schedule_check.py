
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
from planning_guard import PlanningGuard as NumpyGuard
from empty_bounded_geometry import TemporalOccupancy
from runtime_configuration import load_vehicle_config
from occupancy_protocol import unpack
from empty_controller import integrate_state,sweep_allowance
v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';rows=[]
for f in sorted((HERE/'schedule_replays').glob('step_*_result.json')):
 result=_hugsim_expand(json.loads(f.read_text()));selected=result.get('selected')
 if selected is None:continue
 folder=source/f.name.replace('_result.json','')
 with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
 report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));states=np.asarray(selected['states']);times=np.arange(len(states))*v.dt;controls=np.asarray(selected['controls']);interval=selected['control_interval_s'];phase=selected['control_phase_s'];off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];independent=[np.r_[-off,0.,info['ego_velo'],info['ego_steer']]];actions=[]
 for k in range(len(states)-1):
  t=k*v.dt+phase
  if selected['control_interpolation']=='constant':action=controls[min(int(np.floor((t+1e-9)/interval)),5)]
  else:action=np.array([np.interp(t,np.arange(6)*interval,controls[:,j]) for j in range(2)])
  independent.append(integrate_state(independent[-1],action,v.dt,v.wheelbase_m));actions.append(action)
 np.testing.assert_allclose(independent,states,rtol=0.,atol=1e-10);actions=np.asarray(actions)
 with np.load(folder/'perception.npz') as saved:occ,_=unpack(saved)
 assert occ is not None
 depth=report.get('observed_depth_geometry',{});unknown=report.get('unknown_visibility_geometry',{});rotation=info['explicit_l2g_r_mat'];voxel=np.isclose(depth.get('voxel_resolution_m',-1),.5)
 temporal=TemporalOccupancy(np.asarray(occ).reshape(1,5,200,200),v,observed_points=report.get('observed_depth_obstacles_xy'),observed_point_radius=depth.get('voxel_half_diagonal_m',np.sqrt(2)*.25),observed_cell_rotation=rotation if depth.get('policy')=='observed_depth_geometry_with_free_space_clearing_v1' and voxel else None,unknown_points=report.get('unknown_visibility_obstacles_xy'),unknown_point_radius=unknown.get('point_radius_m',np.sqrt(2)*.25),unknown_cell_rotation=rotation if unknown.get('policy')=='endpoint_supported_unknown_visibility_v2' and voxel else None)
 braking=[states[0].copy()]
 for k in range(len(states)-1):
  acc=0. if k*v.dt<report.get('config',{}).get('confirmation_seconds',.2)-1e-9 else -min(v.emergency_braking_mps2,braking[-1][3]/v.dt);braking.append(integrate_state(braking[-1],[acc,0.],v.dt,v.wheelbase_m))
 guard=PlanningGuard(report,v,np.asarray(braking),times,current_occupancy=np.asarray(occ).reshape(5,200,200)[0]);sweep=sweep_allowance(states,v);geometry=float((temporal.clearance(states,times)-sweep).min());proposals=float((NumpyGuard.clearance(guard,states,times)-sweep).min());bounds=bool(states[:,3].min()>=-1e-5 and states[:,3].max()<=v.max_speed_mps+1e-5 and np.abs(states[:,4]).max()<=np.deg2rad(v.max_steer_degrees)+1e-5 and actions[:,0].min()>=v.min_acc_mps2-1e-8 and actions[:,0].max()<=v.max_acc_mps2+1e-8 and np.abs(actions[:,1]).max()<=np.deg2rad(v.max_steer_rate_degrees)+1e-8)
 rows.append(dict(step=folder.name,geometry_gap=geometry,proposal_gap=proposals,physical_bounds=bounds,passed=geometry>=-1e-5 and proposals>=-1e-5 and bounds))
(HERE/'independent_schedule_checks.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2));assert rows and all(x['passed'] for x in rows)
