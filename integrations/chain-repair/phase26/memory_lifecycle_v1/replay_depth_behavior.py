
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
import argparse,copy,json,pickle,time,numpy as np
from depth_consistency import DepthConsistencyMemory
from motion_memory import ContractObstacleMemory
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
from depth_ambiguity import DepthAmbiguity
parser=argparse.ArgumentParser();parser.add_argument('--solve',type=int,nargs='*',default=[]);parser.add_argument('--scene',type=int,choices=[551,552],default=551);args=parser.parse_args()
from run_driver import configure
configure();v=load_vehicle_config();source=CR/('phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop' if args.scene==551 else 'phase29/runs/global_evidence_closed552_v2');spec=h.SceneSpec.read(source/'scene_manifest.json');memory=ContractObstacleMemory.from_source(spec.source);depth=DepthConsistencyMemory(v);rows=[];out=HERE/'depth_behavior_replays'/str(args.scene);out.mkdir(parents=True,exist_ok=True);entry_speed=None
for folder in sorted((source/'uniad_worker').glob('step_*')):
 if not all((folder/name).is_file() for name in ['sensor_metadata.pkl','observed_depth.npz','perception.npz','scene_reliability.json']):continue
 index=int(folder.name.split('_')[-1])
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 if entry_speed is None:entry_speed=float(info['ego_velo'])
 obs={}
 for key,name in [('depth','observed_depth.npz'),('depth_variance','observed_depth_variance.npz'),('depth_alpha','observed_depth_alpha.npz'),('surface_depth','surface_depth.npz'),('surface_valid','surface_valid.npz')]:
  if (folder/name).exists():
   with np.load(folder/name) as z:obs[key]={k:z[k].copy() for k in z.files}
 with np.load(folder/'perception.npz') as z:
  if 'occupancy_contract_json' in z:occ,contract=unpack(z)
  else:occ=z['occupancy'].copy();contract=None
  plan=z['trajectory'].copy();boxes=z['geometry_candidate_boxes'].copy() if 'geometry_candidate_boxes' in z else z['candidate_boxes'].copy();scores=z['candidate_scores'].copy();labels=z['candidate_labels'].copy()
 raw=_hugsim_expand(json.loads((folder/'scene_reliability.json').read_text()));fresh=memory.update(raw,boxes,scores,labels,info,current_occupancy=np.asarray(occ).reshape(5,200,200)[0] if occ is not None else None);report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));report.update({k:fresh[k] for k in ['obstacles','obstacle_memory','global_evidence_receipt','geometry_rejections']});report.pop('velocity_evidence_gate_v2_applied',None);points,audit=depth.update(obs,info);old_count=len(report.get('observed_depth_obstacles_xy',[]));report['observed_depth_obstacles_xy']=points.tolist();report['unknown_visibility_obstacles_xy']=audit.pop('unknown_visibility_points_xy',[]);report['unknown_visibility_geometry']=audit['unknown_visibility_geometry'];report['observed_depth_geometry']=audit;report['requested_cruise_speed_mps']=entry_speed;report['cruise_speed_preference_mps']=entry_speed
 (out/(folder.name+'_guard.json')).write_text(json.dumps(report,indent=2)+'\n');row=dict(step=index,old_hard_depth_points=old_count,new_hard_depth_points=len(points),ambiguity_stop_points=len(audit['height_ambiguity_braking_points_xy']),risk_points=audit['ambiguous_or_expired_cells'])
 ambiguity=DepthAmbiguity(report,info,v);initial=np.r_[-np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3],0.,info['ego_velo'],info['ego_steer']]
 if ambiguity.active:row.update(coast_stop_gap_m=ambiguity.gap(initial,np.array([0.,0.])),brake_stop_gap_m=ambiguity.gap(initial,np.array([-min(2.,info['ego_velo']/.1),0.])) )
 if index in args.solve:
  assert contract is not None,'Legacy transport is replay-only, not a fabricated live contract';info['occupancy_contract']=contract;start=time.perf_counter();result=h.optimize(plan,occ,info,reliability=report,vehicle=v,maxiter=120);(out/(folder.name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');selected=result.get('selected');row.update(selected=selected is not None,first_action=(selected or {}).get('controls',[None])[0],minimum_constraint=(selected or {}).get('minimum_constraint'),forecast_recovery=result.get('forecast_recovery'),solve_seconds=time.perf_counter()-start);print(json.dumps(row),flush=True)
 rows.append(row)
summary=dict(scene=args.scene,frames=len(rows),rows=rows,scope='Current depth/motion memory on archived sensor observations. Selected solves are offline fresh problems, not a new closed loop.',new_gpu_executed=False);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(scene=args.scene,frames=len(rows),solves=[row for row in rows if 'selected' in row]),indent=2))
