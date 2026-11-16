"""Post-hoc paired plans, unmodified HUGSIM native metrics and semantic bounds."""

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

import ast,copy,hashlib,importlib.util,json,pickle,math
pass  # Path is provided by the portable boundary above
import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from paths import SCENE_BUNDLE,SCORER
FIELDS=('nc','dac','ttc','c','pdms','rc','hdscore')
CV_TO_SR=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def native_trajectory_export(points,ego_box):
 source=SCORER.parent/'sim_utils.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='traj_transform_to_global');scope={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
 return scope['traj_transform_to_global'](np.asarray(points),ego_box)

def native_aggregate(records,details):
 tree=ast.parse(SCORER.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScoreCalculator');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='calculate');start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='totals' for t in n.targets))
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='score_list')],kwonlyargs=[],kw_defaults=[],defaults=[]);fn=ast.FunctionDef(name='aggregate',args=args,body=method.body[start:],decorator_list=[]);scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SCORER),'exec'),scope);result=scope['aggregate'](SimpleNamespace(data=records[0]),details);scores=dict(result[3],rc=result[1],hdscore=result[2]);return scores

def raw_plan(plan,info,vehicle):
 plan=np.asarray(plan,dtype=float)
 if plan.shape!=(6,2) or not np.isfinite(plan).all():raise ValueError('Raw UniAD plan must be a finite six-node output')
 c=np.asarray(info['model_lidar_frame']['virtual_to_ego']);r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t']);origin=np.r_[-c[:2,3],-c[2,3]]@r.T+t;xy=origin[[1,0]]*np.array([1.,-1.]);forward=r@np.array([0.,1.,0.]);yaw=float(np.arctan2(-forward[0],forward[1]));ego=[*xy.tolist(),float(origin[2]),2*vehicle.half_width_m,2*vehicle.half_length_m,vehicle.body_height_m,yaw]
 # The stock native exporter uses the sixth XY node for the fifth heading,
 # returning five scored nodes. Preserve this behavior and disclose it.
 local=plan[:,[1,0]]*np.array([1.,-1.]);traj=native_trajectory_export(local,ego)
 return dict(time_stamp=float(info['timestamp']),is_key_frame=True,ego_box=ego,planned_traj=dict(traj=traj,timestep=.5),plan_kind='raw_uniad_stock_native_export',source_nn_nodes=6,scored_native_nodes=5,source_nn_horizon_s=3.,scored_native_horizon_s=2.5)

def actor_boxes(tracks,index):
 valid=np.asarray(tracks['visible'])[index];t=np.asarray(tracks['translations'])[index];q=np.asarray(tracks['quaternions_wxyz'])[index];size=np.asarray(tracks['size_lwh']);ids=np.flatnonzero(valid);boxes=[]
 for i in ids:
  rotation=Rotation.from_quat(q[i][[1,2,3,0]]);forward=rotation.as_matrix()[:,0];boxes.append([*map(float,t[i]),float(size[i,1]),float(size[i,0]),float(size[i,2]),float(np.arctan2(forward[1],forward[0]))])
 return boxes,['car']*len(boxes)

def route_completion(xy):
 route=np.load(SCENE_BUNDLE/'native_dense_route.npy',mmap_mode='r');p=np.asarray(route)[:,[0,2],3];native=np.array([-xy[1],xy[0]]);index=int(np.linalg.norm(p-native,axis=1).argmin());return (index+1)/(.9*len(route))

def export(run,vehicle):
 run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));n=len(report.get('trace',[]));tracks=dict(np.load(run/'edited_actor_tracks.npz'));streams={'raw_uniad':{},'adopted_plan':{}};links=[];discarded=[]
 for folder in sorted((run/'uniad_worker').glob('step_*')):
  index=int(folder.name.split('_')[-1]);a=folder/'native_adopted_plan.json';b=folder/'perception.npz';m=folder/'sensor_metadata.pkl'
  if index>=n:
   if a.exists():discarded.append(index)
   continue
  if not all(f.exists() for f in (a,b,m)):raise ValueError('Missing raw/adopted native plan evidence '+str(folder))
  with m.open('rb') as f:info=pickle.load(f)
  with np.load(b) as z:raw=raw_plan(z['trajectory'].copy(),info,vehicle)
  adopted=_hugsim_expand(json.loads(a.read_text()));adopted['planned_traj']['traj']=native_trajectory_export(np.asarray(adopted['planned_traj']['traj'])[:6,:2],[0.,0.,0.,1.,1.,1.,0.])
  if len(adopted['planned_traj']['traj'])!=5 or adopted['planned_traj']['timestep']!=.5 or not np.allclose(adopted['ego_box'],raw['ego_box'],rtol=0,atol=1e-6):raise ValueError('Native raw/adopted frame or horizon mismatch')
  if raw['time_stamp']!=adopted['time_stamp'] or abs(raw['time_stamp']-index/10.)>1e-7:raise ValueError('Score clock mismatch')
  for kind,value in [('raw_uniad',raw),('adopted_plan',adopted)]:streams[kind][index]=value
  links.append(dict(step=index,raw_npz_sha256=digest(b),metadata_sha256=digest(m),adopted_sha256=digest(a)))
 if not links:raise ValueError('No confirmed executed-control plan pair to score')
 final=max(streams['raw_uniad'])+25
 if final>=len(tracks['translations']):raise ValueError('Full future actor support unavailable; never hold last actor state to fill score horizon')
 final_rc=max(route_completion(v['ego_box'][:2]) for v in streams['raw_uniad'].values());data={}
 for kind,keyframes in streams.items():
  frames=[];last=next(iter(keyframes.values()))
  for i in range(final+1):
   value=copy.deepcopy(keyframes[i]) if i in keyframes else dict(time_stamp=i/10.,is_key_frame=False,ego_box=last['ego_box'].copy(),planned_traj=dict(traj=[],timestep=.5),role='future_actor_support_only')
   if i in keyframes:last=value
   value['obj_boxes'],value['obj_names']=actor_boxes(tracks,i)
   # Future-only actor support must never invent additional ego progress.
   value['rc']=route_completion(value['ego_box'][:2]) if i in keyframes else final_rc;frames.append(value)
  data[kind]=[dict(frames=frames,metas=dict(plan_kind=kind,paired_same_observations=True,independent_raw_policy_closed_loop=False))]
 folder=run/'native_eval';folder.mkdir(exist_ok=True)
 for kind,value in data.items():
  with (folder/(kind+'_data.pkl')).open('wb') as f:pickle.dump(value,f,protocol=4)
 manifest=dict(schema='native_paired_plan_score_input_v1',native_source_sha256=digest(SCORER),scenario_sha256=digest(__import__('paths').HERE/'scenario.json'),source_scene_manifest_sha256=digest(SCENE_BUNDLE/'source_manifest.json'),ego_geometry_sha256=hashlib.sha256(json.dumps([vehicle.half_width_m,vehicle.half_length_m,vehicle.body_height_m]).encode()).hexdigest(),uniad_seed=report.get('uniad_seed'),uniad_weights_sha256=report.get('checkpoint_before',{}).get('sha256'),static_background_sha256=digest(SCENE_BUNDLE/'background_sr_world.npy'),static_road_sha256=digest(SCENE_BUNDLE/'road_sr_world.npy'),point_semantics_sha256=digest(SCENE_BUNDLE/'point_semantics/background_semantics.npz'),actor_values_sha256=hashlib.sha256(b''.join(str((k,tracks[k].dtype.str,tracks[k].shape)).encode()+np.ascontiguousarray(tracks[k]).tobytes() for k in sorted(tracks))).hexdigest(),pairs=len(links),confirmed_controls=n,discarded_unconfirmed_attempt_steps=discarded,future_actor_last_frame=final,plan_horizon_s=2.5,source_nn_horizon_s=3.,native_export_helper_sha256=digest(SCORER.parent/'sim_utils.py'),raw_xy_prediction_modified=False,adopted_plan_truncated_to_common_horizon=True,ground_truth_used_by_optimizer=False,raw_plan_is_independent_closed_loop=False,links=links,episode_complete=bool(n==100 and report.get('status')=='completed'),episode_termination=report.get('termination_reason',report.get('error')),actor_tracks_sha256=digest(run/'edited_actor_tracks.npz'),scoring_scope='Unmodified native algorithm on edited Framework reconstruction; semantic uncertainty bounded, not a formal benchmark certificate')
 (folder/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return data,manifest

def score(run,vehicle):
 run=Path(run);data,manifest=export(run,vehicle);out=run/'native_eval';result=dict(status='gpu_pending',comparison_key={k:manifest[k] for k in ('native_source_sha256','scenario_sha256','source_scene_manifest_sha256','uniad_seed','plan_horizon_s','native_export_helper_sha256','uniad_weights_sha256','static_background_sha256','static_road_sha256','point_semantics_sha256','actor_values_sha256','ego_geometry_sha256')},native_source_sha256=manifest['native_source_sha256'],scores=None,formal_evaluation_admitted=False,ground_truth_used_by_optimizer=False,raw_baseline_is_counterfactual_plan=True)
 import torch
 if not torch.cuda.is_available():
  result['reason']='CUDA required by unmodified HUGSIM native scorer';(out/'native_scores.json').write_text(json.dumps(result,indent=2)+'\n');return result
 spec=importlib.util.spec_from_file_location('official_hugsim_native_score',SCORER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 background=np.load(SCENE_BUNDLE/'background_sr_world.npy');road=np.load(SCENE_BUNDLE/'road_sr_world.npy')
 with np.load(SCENE_BUNDLE/'point_semantics/background_semantics.npz') as z:labels=z['class_id'].copy()
 if len(labels)!=len(background):raise ValueError('Semantic source/geometry mismatch')
 known=background[(labels>1)&(labels!=10)&(labels!=255)];unknown=background[labels==255];ground=np.concatenate([road,background[labels==0]])
 result.update(status='computed_native_metric_bounds',scores={},unknown_background_points=len(unknown),total_background_points=len(background),semantic_coverage_complete=len(unknown)==0,weights_unchanged=True,native_code_unmodified=True,paired_same_observations=True,episode_complete=manifest['episode_complete'])
 from scipy.spatial import cKDTree
 clouds={'known':known,'unknown':unknown,'ground':ground};trees={name:cKDTree(values[:,:2]) for name,values in clouds.items()};indices={kind:{i:f for i,f in enumerate(records[0]['frames']) if f['is_key_frame']} for kind,records in data.items()};all_scores={kind:{bound:[] for bound in ('pessimistic','optimistic')} for kind in data};details={kind:{bound:{} for bound in ('pessimistic','optimistic')} for kind in data}
 for frame_index in indices['raw_uniad']:
  # Crop only points outside a conservative common query ROI. No thinning,
  # voting-density change or per-plan semantic difference is introduced.
  queries=[];radius=0.
  for kind in data:
   frame=indices[kind][frame_index];traj=np.vstack([np.array(frame['ego_box'])[[0,1,6]],np.asarray(frame['planned_traj']['traj'])]);xy=traj[:,:2];vel=np.diff(xy,axis=0)/.5;vel=np.vstack([vel[0],vel]);queries.extend([xy,xy+vel*.5,xy+vel]);radius=max(radius,float(np.hypot(frame['ego_box'][3],frame['ego_box'][4])))
  positions=np.concatenate(queries);low=positions.min(0)-radius-.01;high=positions.max(0)+radius+.01;center=(low+high)/2;extent=np.linalg.norm(high-low)/2+.01;local={}
  for name,values in clouds.items():
   ids=np.asarray(trees[name].query_ball_point(center,extent),dtype=int);selected=values[ids];local[name]=selected[np.all((selected[:,:2]>=low)&(selected[:,:2]<=high),axis=1)]
  for kind,records in data.items():
   # All future boxes are genuine programmed/source-clock states. Frame data
   # outside this keyframe is actor support only; route completion stays fixed.
   frames=copy.deepcopy(records[0]['frames'][frame_index:frame_index+26]);frames[0]['is_key_frame']=True
   for frame in frames[1:]:frame['is_key_frame']=False
   for bound in ('pessimistic','optimistic'):
    bg=np.concatenate([local['known'],local['unknown']]) if bound=='pessimistic' else local['known'];g=local['ground'] if bound=='pessimistic' else np.concatenate([local['ground'],local['unknown']])
    value=module.hugsim_evaluate([dict(frames=copy.deepcopy(frames),metas=records[0].get('metas',{}))],np.asarray(g@CV_TO_SR,dtype=np.float32),np.asarray(bg@CV_TO_SR,dtype=np.float32))
    clean={key:float(value[key]) for key in FIELDS}
    if not all(np.isfinite(x) and 0<=x<=1.000001 for x in clean.values()):raise ValueError('Invalid native score')
    all_scores[kind][bound].append(clean);details[kind][bound][str(frames[0]['time_stamp'])]={k:clean[k] for k in ('nc','dac','ttc','c','pdms')}
 for kind in data:
  bounds={bound:native_aggregate(records=data[kind],details={float(t):v for t,v in details[kind][bound].items()}) for bound in ('pessimistic','optimistic')}
  for bound in bounds:(out/(kind+'_'+bound+'_native_details.json')).write_text(json.dumps(dict(scores=bounds[bound],details=details[kind][bound]),indent=2)+'\n')
  for key in FIELDS:
   if bounds['pessimistic'][key]>bounds['optimistic'][key]+1e-7:raise ValueError('Non-monotone native score bounds')
  result['scores'][kind]={key:dict(lower=bounds['pessimistic'][key],upper=bounds['optimistic'][key],value=bounds['pessimistic'][key] if bounds['pessimistic'][key]==bounds['optimistic'][key] else None) for key in FIELDS}
 result['point_selection']='Conservative common per-frame ROI including all native TTC projections; full point density retained';result['native_frame_aggregation']='Frame-independent native metrics averaged; future-only RC equals completed episode progress'
 result['robust_pdms_improvement_on_paired_plans']=result['scores']['adopted_plan']['pdms']['lower']>result['scores']['raw_uniad']['pdms']['upper']
 result['single_value_score_available']=all(v['lower']==v['upper'] for stream in result['scores'].values() for v in stream.values());result['formal_native_score_available']=False;(out/'native_scores.json').write_text(json.dumps(result,indent=2)+'\n');return result
