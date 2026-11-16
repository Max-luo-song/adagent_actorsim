
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
from runtime_configuration import load_vehicle_config
from evidence_guard import PlanningGuard
from planning_guard import PlanningGuard as NumpyGuard
from empty_bounded_geometry import TemporalOccupancy
from global_repair_v2 import controller as old
SOURCE=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop'
rows=[]
for index in (46,100):
 folder=SOURCE/f'uniad_worker/step_{index:03d}'
 with (folder/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
 del obs
 report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));v=load_vehicle_config();result=_hugsim_expand(json.loads((HERE/f'current_replays/step_{index:03d}_result.json').read_text()));selected=result['selected'];off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
 states,_=old.rollout(selected['controls'],info['ego_velo'],info['ego_steer'],off,v.dt,v.wheelbase_m,selected['control_interval_s'],interpolation=report.get('control_interpolation','constant'));np.testing.assert_allclose(states,selected['states'],rtol=0,atol=1e-10);times=np.arange(len(states))*v.dt;sweep=old.sweep_allowance(states,v)
 with np.load(folder/'perception.npz') as saved:occ=saved['occupancy'].reshape(1,5,200,200);plan=saved['trajectory'].copy()
 temporal=TemporalOccupancy(occ,v,observed_points=report['observed_depth_obstacles_xy'],observed_cell_rotation=info['explicit_l2g_r_mat'],observed_point_radius=report['observed_depth_geometry']['voxel_half_diagonal_m'],unknown_points=report['unknown_visibility_obstacles_xy'],unknown_cell_rotation=info['explicit_l2g_r_mat'],unknown_point_radius=report['unknown_visibility_geometry']['point_radius_m'])
 gap=float((temporal.clearance(states,np.zeros(len(states)) if result.get('forecast_recovery') else times)-sweep).min());braking=[np.r_[-off,0.,info['ego_velo'],info['ego_steer']]]
 for k in range(len(states)-1):
  acc=0. if k*v.dt<report['config']['confirmation_seconds']-1e-9 else -min(v.emergency_braking_mps2,braking[-1][3]/v.dt)
  braking.append(old.integrate_state(braking[-1],[acc,0.],v.dt,v.wheelbase_m))
 guard=PlanningGuard(report,v,np.asarray(braking),times,current_occupancy=occ[0,0]);proposal_gap=float((NumpyGuard.clearance(guard,states,times)-sweep).min());controls=np.asarray(selected['controls'])
 bounds=bool(states[:,3].min()>=-1e-5 and states[:,3].max()<=v.max_speed_mps+1e-5 and np.abs(states[:,4]).max()<=np.deg2rad(v.max_steer_degrees)+1e-5 and controls[:,0].min()>=v.min_acc_mps2-1e-8 and controls[:,0].max()<=v.max_acc_mps2+1e-8 and np.abs(controls[:,1]).max()<=np.deg2rad(v.max_steer_rate_degrees)+1e-8)
 nav=result['guard']['navigation_reference'];np.testing.assert_array_equal(nav['original_neural_reference'],plan);np.testing.assert_array_equal(np.asarray(nav['tracking_reference'])[:6],plan)
 rows.append(dict(step=index,passed=bool(gap>=-1e-5 and proposal_gap>=-1e-5 and bounds),geometry_swept_gap_m=gap,proposal_swept_gap_m=proposal_gap,physical_bounds_passed=bounds,current_frame_reference_exact=True,road_soft_cost_weight=result['guard']['road_soft_cost_weight']))
(HERE/'independent_current_checks.json').write_text(json.dumps(rows,indent=2)+chr(10));print(json.dumps(rows,indent=2));assert all(r['passed'] for r in rows)
