
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
from empty_bounded_geometry import TemporalOccupancy
from evidence_guard import PlanningGuard
from planning_guard import PlanningGuard as NumpyGuard
from global_repair_v2 import controller as previous
from runtime_configuration import load_vehicle_config
SOURCE=CR/'phase31/runs/depth_consistency551_20261105T051937Z_fc796426/closed_loop'
rows=[]
for i in (40,70):
 f=SOURCE/f'uniad_worker/step_{i:03d}'
 with (f/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
 del obs
 rel=_hugsim_expand(json.loads((f/'guard_memory_report.json').read_text()));v=load_vehicle_config();result=_hugsim_expand(json.loads((HERE/f'replays/step_{i:03d}_result.json').read_text()));chosen=result['selected'];off=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
 states,_=previous.rollout(chosen['controls'],info['ego_velo'],info['ego_steer'],off,v.dt,v.wheelbase_m,chosen['control_interval_s'],interpolation=rel.get('control_interpolation','constant'));np.testing.assert_allclose(states,chosen['states'],rtol=0,atol=1e-10);times=np.arange(len(states))*v.dt;sweep=previous.sweep_allowance(states,v)
 occ=np.zeros((1,5,200,200));temporal=TemporalOccupancy(occ,v,observed_points=rel['observed_depth_obstacles_xy'],observed_cell_rotation=info['explicit_l2g_r_mat'],observed_point_radius=rel['observed_depth_geometry']['voxel_half_diagonal_m'],unknown_points=rel['unknown_visibility_obstacles_xy'],unknown_cell_rotation=info['explicit_l2g_r_mat'],unknown_point_radius=rel['unknown_visibility_geometry']['point_radius_m'])
 gap=float((temporal.clearance(states,np.zeros(len(states)) if result.get('forecast_recovery') else times)-sweep).min())
 braking=[np.r_[-off,0.,info['ego_velo'],info['ego_steer']]]
 for k in range(60):
  acc=0. if k*v.dt<rel['config']['confirmation_seconds']-1e-9 else -min(v.emergency_braking_mps2,braking[-1][3]/v.dt)
  braking.append(previous.integrate_state(braking[-1],[acc,0.],v.dt,v.wheelbase_m))
 guard=PlanningGuard(rel,v,np.asarray(braking),times,current_occupancy=occ[0,0]);proposal=float((NumpyGuard.clearance(guard,states,times)-sweep).min());cap=result['guard'].get('speed_limit_mps');cap_ok=cap is None or bool(np.min(np.maximum(float(cap),float(info['ego_velo'])-v.emergency_braking_mps2*times)-states[:,3])>=-1e-5)
 rows.append(dict(step=i,passed=bool(gap>=-1e-5 and proposal>=-1e-5 and cap_ok),independent_numpy_geometry_gap_m=gap,independent_numpy_proposal_gap_m=proposal,reachable_caution_cap_passed=cap_ok))
(HERE/'independent_checks.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2));assert all(r['passed'] for r in rows)
