"""Post-hoc proxy test only. Never imported by the driver/controller loop."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import argparse,json,pickle
pass  # Path is provided by the portable boundary above
import numpy as np

def footprint(pose,half_length,half_width):
 local=np.array([[x,y,0,1] for x,y in [(-half_length,-half_width),(half_length,-half_width),(half_length,half_width),(-half_length,half_width)]])
 return (np.asarray(pose)@local.T).T[:,:2]

def evaluate_rows(observations,obstacle,vehicle,*,hold_seconds=1.,route_tolerance=1.,yaw_tolerance=.1):
 first=np.asarray(observations[0]['closed_ego_to_world']);forward=first[:2,0];forward=forward/np.linalg.norm(forward);end=float((np.asarray(obstacle)@forward).max())
 route=np.array([row['open_ego_to_world'] for row in observations])[:,:2,3];segments=np.diff(route,axis=0);length2=np.sum(segments**2,axis=1);rows=[];since=None;returned=False
 for row in observations:
  pose=np.asarray(row['closed_ego_to_world']);center=pose[:2,3];body=footprint(pose,vehicle['half_length_m'],vehicle['half_width_m']);rear=float((body@forward).min());passed=rear>end+vehicle['clearance_margin_m']
  if len(segments):
   u=np.clip(np.sum((center-route[:-1])*segments,axis=1)/np.maximum(length2,1e-12),0,1);projected=route[:-1]+u[:,None]*segments;dist=np.linalg.norm(center-projected,axis=1);index=int(np.argmin(dist));error=float(dist[index]);direction=segments[index]/max(np.sqrt(length2[index]),1e-12);heading=pose[:2,0];yaw=float(abs(np.arctan2(heading[0]*direction[1]-heading[1]*direction[0],heading@direction)))
  else:error=float(np.linalg.norm(center-route[0]));yaw=0.
  aligned=passed and error<=route_tolerance and yaw<=yaw_tolerance and row['closed_speed_mps']>=.5
  since=row['timestamp'] if aligned and since is None else since if aligned else None
  if since is not None and row['timestamp']-since>=hold_seconds-1e-8:returned=True
  overlap=bool(row.get('native_vehicle_reference',{}).get('overlap') or row.get('all_actor_reference',{}).get('overlap') or row.get('inserted_reference_gap_m',float('inf'))<=0)
  rows.append(dict(timestamp=row['timestamp'],source_frame=row['source_frame'],passed_entire_footprint=passed,route_error_m=error,heading_error_rad=yaw,proxy_overlap=overlap,reference_gap_m=row.get('inserted_reference_gap_m'),speed_mps=row['closed_speed_mps']))
 return dict(passed_obstacle=any(x['passed_entire_footprint'] for x in rows),returned_to_recorded_route=returned,no_reported_proxy_overlap=not any(x['proxy_overlap'] for x in rows),avoidance_proxy_success=returned and not any(x['proxy_overlap'] for x in rows),criteria=dict(entire_body_past_obstacle=True,margin_m=vehicle['clearance_margin_m'],route_tolerance_m=route_tolerance,yaw_tolerance_rad=yaw_tolerance,hold_seconds=hold_seconds),rows=rows,formal_evaluation_admitted=False,reference_only=True,ground_truth_used_by_controller=False)

def analyze(run):
 run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));ref=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()));manifest=_hugsim_expand(json.loads((run/'scene_manifest.json').read_text()));vehicle=_hugsim_expand(json.loads(Path(manifest['vehicle_config']).read_text()));pose=np.array(ref['placements'][0]['asset_to_source']);width,length,_=ref['asset_wlh'];local=np.array([[x,ref['local_bottom_y'],z,1] for x,z in [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]]);obstacle=(pose@local.T).T[:,:2]
 result=evaluate_rows(report['observations'],obstacle,vehicle);result.update(driving_status=report.get('status'),observations=len(report['observations']),requested_controls=report.get('steps_requested'),scope='reconstruction/calibration box proxies, not contact certification',planned_maneuvers=[],target_proximity_diagnostics=[])
 error=str(report.get('error',''))+' '+str(report.get('termination_reason',''))
 if any(word in error.lower() for word in ('overlap','collision')):result['no_reported_proxy_overlap']=False
 result['avoidance_proxy_success']=result['avoidance_proxy_success'] and result['no_reported_proxy_overlap'] and report.get('status')=='completed'
 result['outcome']='passed_and_returned_proxy' if result['avoidance_proxy_success'] else 'proxy_collision_or_overlap' if not result['no_reported_proxy_overlap'] else 'passed_without_route_return' if result['passed_obstacle'] else 'not_passed_within_observed_window'
 B=np.array([[1,0,0],[0,0,1],[0,-1,0]]);world=B@np.array(ref['placements'][0]['asset_to_native'])[:3,3]
 for index in sorted({0,max(0,len(report['trace'])//2),max(0,len(report['trace'])-1)}):
  folder=run/f'uniad_worker/step_{index:03d}';op=folder/'optimization.json'
  if not op.exists():continue
  data=_hugsim_expand(json.loads(op.read_text()));selected=data.get('selected') or {};states=np.array(selected.get('states',[]))
  if len(states):result['planned_maneuvers'].append(dict(step=index,planned_lateral_min_m=float(states[:,0].min()),planned_lateral_max_m=float(states[:,0].max()),planned_forward_m=float(states[-1,1]-states[0,1]),minimum_constraint=selected.get('minimum_constraint')))
  try:
   with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
   local=(world-np.array(info['explicit_l2g_t']))@np.array(info['explicit_l2g_r_mat'])
   with np.load(folder/'perception.npz') as f:
    boxes=f['candidate_boxes'].reshape(-1,9);scores=f['candidate_scores'].reshape(-1);j=int(np.argmin(np.linalg.norm(boxes[:,:2]-local[:2],axis=1)));distance=float(np.linalg.norm(boxes[j,:2]-local[:2]))
    result['target_proximity_diagnostics'].append(dict(step=index,nearest_candidate_distance_m=distance,nearest_candidate_score=float(scores[j]),nearest_candidate_velocity_xy=boxes[j,7:9].tolist(),identity_confirmed=False,association_is_posthoc_proximity_only=True))
  except (FileNotFoundError,KeyError):pass
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();result=analyze(args.run);out=args.run/'analysis';out.mkdir(exist_ok=True);(out/'avoidance_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
