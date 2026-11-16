
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
import argparse,copy,json,pickle,numpy as np
from planning_guard import PlanningGuard as NumpyGuard,rectangle_separation
from speed_policy import PlanningGuard
from time_local_motion import tube_boxes
from empty_bounded_geometry import TemporalOccupancy,polygon_clearance,projected_cell
from empty_controller import integrate_state,sweep_allowance
from depth_ambiguity import DepthAmbiguity
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,nargs='+',default=[0]);args=parser.parse_args();v=load_vehicle_config();source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker';out=HERE/'depth_behavior_replays/551';rows=[]
def proposal(guard,states,times,full=False):
 keep=np.array([i for i in range(len(guard.hard_boxes)) if i not in guard.local_motion],dtype=int);remaining=copy.copy(guard);names=('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges')
 for name in names:setattr(remaining,name,getattr(guard,name)[keep])
 gaps=NumpyGuard.clearance(remaining,states,times)
 for i,cert in guard.local_motion.items():
  boxes=tube_boxes(guard.hard_boxes[i],cert,times,guard.hard_radii[i]);gap=np.diag(rectangle_separation(states,boxes,np.zeros(len(times)),v.half_width_m,v.half_length_m))-guard.hard_margins[i];single=copy.copy(guard)
  for name in names:setattr(single,name,getattr(guard,name)[i:i+1])
  gap[times==0]=NumpyGuard.clearance(single,states,times)[times==0]
  if not full:gap[times>guard.motion_lease+1e-9]=np.inf
  gaps=np.minimum(gaps,gap)
 return gaps
for index in args.steps:
 name='step_%03d'%index;folder=source/name;result=_hugsim_expand(json.loads((out/(name+'_result.json')).read_text()));selected=result.get('selected');assert selected is not None,'No selected plan for '+name;assert result['forecast_recovery'] is False
 report=_hugsim_expand(json.loads((out/(name+'_guard.json')).read_text()))
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 with np.load(folder/'perception.npz') as z:occ,_=unpack(z)
 states=np.asarray(selected['states']);controls=np.asarray(selected['controls']);times=np.arange(len(states))*v.dt;interval=selected['control_interval_s'];offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3];independent=[np.r_[-offset,0.,info['ego_velo'],info['ego_steer']]];actions=[]
 for k in range(len(states)-1):
  if result['guard']['control_interpolation']=='constant':action=controls[min(int(np.floor((k*v.dt+1e-9)/interval)),5)]
  else:action=np.array([np.interp(k*v.dt,np.arange(6)*interval,controls[:,j]) for j in range(2)])
  independent.append(integrate_state(independent[-1],action,v.dt,v.wheelbase_m));actions.append(action)
 np.testing.assert_allclose(independent,states,rtol=0.,atol=1e-10)
 depth=report['observed_depth_geometry'];unknown=report['unknown_visibility_geometry'];r=info['explicit_l2g_r_mat'];voxel=np.isclose(depth.get('voxel_resolution_m',-1),.5)
 temporal=TemporalOccupancy(np.asarray(occ).reshape(1,5,200,200),v,observed_points=report['observed_depth_obstacles_xy'],observed_point_radius=depth['voxel_half_diagonal_m'],observed_cell_rotation=r if voxel else None,unknown_points=report['unknown_visibility_obstacles_xy'],unknown_point_radius=unknown['point_radius_m'],unknown_cell_rotation=r if unknown.get('policy')=='endpoint_supported_unknown_visibility_v2' and voxel else None)
 braking=DepthAmbiguity(report,info,v).stop_states(states[0],np.array([-min(2.,states[0,3]/.1),0.]))
 guard=PlanningGuard(report,v,braking,np.arange(len(braking))*.1,current_occupancy=np.asarray(occ).reshape(5,200,200)[0])
 def clear(query,full=False):
  clock=np.arange(len(query))*.1;return np.minimum(temporal.clearance(query,clock),proposal(guard,query,clock,full=full))-sweep_allowance(query,v)
 def physical(query):
  acc=np.diff(query[:,3])/.1;rate=np.diff(query[:,4])/.1
  return bool(np.isfinite(query).all() and query[:,3].min()>=-1e-5 and query[:,3].max()<=v.max_speed_mps+1e-5 and np.abs(query[:,4]).max()<=np.deg2rad(v.max_steer_degrees)+1e-5 and acc.min()>=v.min_acc_mps2-1e-5 and acc.max()<=v.max_acc_mps2+1e-5 and np.abs(rate).max()<=np.deg2rad(v.max_steer_rate_degrees)+1e-5)
 budget=DepthAmbiguity(report,info,v);witnesses=[]
 for mode in ['hold','unwind','left','right']:
  query=budget.stop_states(states[0],np.asarray(actions[0]),mode);expanded=np.c_[query,np.full(len(query),v.half_width_m),np.full(len(query),v.half_length_m)];gaps=polygon_clearance(expanded,budget.points,projected_cell(r),v.clearance_margin_m+1e-7)-sweep_allowance(query,v);gap=float(np.minimum(gaps,clear(query,full=True)).min());witnesses.append(dict(mode=mode,gap_m=gap,physical_bounds=physical(query),states=query.tolist()))
 best=max(witnesses,key=lambda row:row['gap_m']);gap=float(clear(states).min());row=dict(step=index,first_action=actions[0].tolist(),minimum_hard_gap_m=gap,physical_bounds=physical(states),braking_witness=best,passed=gap>=-1e-5 and physical(states) and best['gap_m']>=-1e-5 and best['physical_bounds'],future_ambiguous_surface_path_certified=False);rows.append(row)
 (out/(name+'_independent_check.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='braking_witness'}|{'braking_mode':best['mode'],'braking_gap_m':best['gap_m']},indent=2))
(out/'independent_checks.json').write_text(json.dumps(dict(passed=all(row['passed'] for row in rows),rows=rows,new_gpu_executed=False),indent=2)+'\n');assert all(row['passed'] for row in rows)
