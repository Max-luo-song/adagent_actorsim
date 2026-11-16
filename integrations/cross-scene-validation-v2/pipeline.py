"""Fresh cross-scene off/on episodes on explicitly admitted reconstruction-support segments."""

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

import argparse,copy,fcntl,hashlib,json,os,subprocess,sys,time,traceback
pass  # Path is provided by the portable boundary above
import numpy as np
from shapely.geometry import shape,Polygon
from bindings import ROOT,LOCAL,CANON,OLD,MAP,bind
CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair';P26=CR/'phase26';OUT=P26/'cross_scene_validation_v2'
ASSETS=P26/'patch_generalization_assets_v1';REGISTRY=P26/'autonomous_patch_agent_v7/campaigns/patch_auto_v7/candidate_registry.json'
REPORTS={172:ROOT/'outputs/waymo172_three_setup_20261016_132050_295391/source_report.json',552:CR/'phase4/dynamic552_frame80/source_report.json',621:ROOT/'outputs/avoidance_scene_search_20261022/scene621/source_report.json'}
COMMANDS={172:1,552:0,621:2};ARMS=['off','v15','v17'];TEMPLATE=P26/'light_tracking_curvature_v15_r2'
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Frozen asset changed: '+a['path'])
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def source_pins():
 paths=[*LOCAL.glob('*.py'),LOCAL/'run.sh',OLD/'common.py',OLD/'environment.py',OLD/'tracker.py',OLD/'diagnostics.py',MAP/'map_reference.py',MAP/'scoring.py',MAP/'local_boot.py',ROOT/'integrations/patch-v15-validation-r2-v1/controller_switch.py',ROOT/'integrations/sr-hugsim/uniad_bridge.py',ROOT/'integrations/sr-hugsim/scene_api.py',ROOT/'waymo23-native-import/configs/vehicle.json',ROOT/'code/HUGSIM/sim/utils/score_calculator.py',ROOT/'code/HUGSIM/sim/utils/sim_utils.py',CR/'phase46/cutin551_white_edge_v1/native_car_edit.py',CR/'phase46/cutin551_white_edge_v1/native_scores.py']
 for code in [ROOT/'integrations/light-tracking-curvature-v15-r2',CANON]:paths.extend(code.rglob('*.py'));paths.append(code/'run.sh')
 inherited=read(TEMPLATE/'plan.json')['code']
 for path,h in inherited.items():
  if sha(path)!=h:raise ValueError('Inherited frozen renderer/model dependency changed: '+path)
 return inherited|{str(p.resolve()):sha(p) for p in set(paths)}
def verify_pins(pins):
 for p,h in pins.items():
  if sha(p)!=h:raise ValueError('Code/physics/scorer changed: '+p)
def choose_start(geometry,poses,v,min_controls=30):
 corners=np.array([[-v['half_length_m'],-v['half_width_m'],0],[v['half_length_m'],-v['half_width_m'],0],[v['half_length_m'],v['half_width_m'],0],[-v['half_length_m'],v['half_width_m'],0]])
 for i in range(10,len(poses)-min_controls):
  p=poses[i];body=Polygon((corners@p[:3,:3].T+p[:3,3])[:,:2])
  speed=float(np.linalg.norm(poses[i+1,:3,3]-poses[i,:3,3])/.1)
  stop_distance=speed*speed/(-2*v.get('min_acc_mps2',-2.))+.1*speed
  tube=corners.copy();tube[tube[:,0]>0,0]+=stop_distance
  stopping_body=Polygon((tube@p[:3,:3].T+p[:3,3])[:,:2])
  reserve=v['clearance_margin_m']+.3+.05+.05*speed
  if geometry.covers(stopping_body) and stopping_body.distance(geometry.boundary)>=reserve:return i
 return None

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 previous=P26/'cross_scene_validation_v1/started.json'
 if previous.exists() and not (OUT/'started.json').exists():
  shutil_data=read(previous);shutil_data['inherited_from']=asset(previous);write(OUT/'started.json',shutil_data)
 if (OUT/'matrix.json').exists():
  m=read(OUT/'matrix.json');verify_pins(m['pins'])
  for a in m['inputs']:verify(a)
  return m
 registry=read(REGISTRY);valid={k:v for k,v in registry.items() if all(v.get('metrics',{}).get(c,{}).get('status')=='completed' for c in ['original551','cutin551'])}
 best=max(valid,key=lambda k:valid[k]['metrics']['cutin551']['score']['hdscore'])
 if best!='v15':raise ValueError('Best measured branch changed; register a new named benchmark')
 v=read(ROOT/'waymo23-native-import/configs/vehicle.json');template=read(TEMPLATE/'plan.json');cases={};write(OUT/'registry_snapshot.json',registry);inputs=[asset(OUT/'registry_snapshot.json')]
 for scene in [552,172,621]:
  bundle=ASSETS/f'scene{scene}/bundle';meta=read(bundle/'source_manifest.json');raw=read(bundle/'raw_metadata/raw_metadata.json');sm=read(bundle/'semantic19/semantic_manifest.json');pm=read(bundle/'point_semantics/point_semantics_manifest.json');support=read(bundle/'map_support/manifest.json');geo_path=bundle/'map_support/road_support.geojson'
  if meta['scene']!=scene or raw['scene_index']!=scene or len(raw['frames'])!=meta['frames'] or len(sm['maps'])!=meta['frames']*5 or len(pm['frames'])!=meta['frames'] or pm['unknown_is_free']:raise ValueError('Incomplete/mismatched asset coverage')
  for name,h in meta['artifacts'].items():
   if sha(bundle/name)!=h:raise ValueError('Changed geometry export')
  if sha(meta['source_checkpoint'])!=meta['source_checkpoint_sha256']:raise ValueError('Changed trained scene')
  verify(support['source']);verify(support['geometry']);geojson=read(geo_path)
  if not geojson['properties']['unknown_holes_preserved'] or geojson['properties']['hole_filling']:raise ValueError('Unknown road region filled')
  geometry=shape(geojson['geometry']);poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];start=choose_start(geometry,poses,v)
  inputs.extend(asset(bundle/n) for n in ['source_manifest.json','raw_metadata/raw_metadata.json','semantic19/semantic_manifest.json','point_semantics/point_semantics_manifest.json','point_semantics/background_semantics.npz','map_support/road_support.geojson'])
  inputs.extend(asset(p) for p in [Path(meta['source_checkpoint']),REPORTS[scene]])
  if start is None:
   cases[str(scene)]={'scene':scene,'admitted':False,'reason':'No fully observed initial body footprint with 30 controls remaining; no borrowed map, no artificial free-space filling','score_deltas':None};continue
  speed=float(np.linalg.norm(poses[start+1,:3,3]-poses[start,:3,3])/.1);controls=len(poses)-1-start
  initial={'source_frame':start,'ego_to_world':poses[start].tolist(),'initial_speed_mps':speed}
  map_data={'schema':'reconstructed_road_support_map_v1','authority':'reconstructed_support_proxy','geometry':geojson['geometry'],'holes_preserved':True,'uncertainty_m':.3,'online_static_map_allowed':True,'physical_white_boundary_certified':False,'source':asset(geo_path),'scope':'conservative observed reconstruction support, not surveyed lane/white-line boundaries'}
  scene_row={'scene':scene,'admitted':True,'start_frame':start,'controls':controls,'navigation_command':COMMANDS[scene],'full_original_scene_test':False,'scope':'coverage-selected source segment; map-proxy sensitivity experiment','arms':{}}
  for arm in ARMS:
   work=OUT/'cases'/str(scene)/arm
   if work.exists():raise ValueError('Partial preparation retained; inspect before creating a new named benchmark')
   work.mkdir(parents=True)
   candidate=registry['v17' if arm=='v17' else best]
   for p,h in candidate['code_pins'].items():
    if sha(p)!=h:raise ValueError('Controller differs from measured release')
   for name in ['baseline_freeze.json','baseline_verification.json','boot/manifest.json']:
    dst=work/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((TEMPLATE/name).read_bytes())
   manifest={'schema':'unified_scene_v1','backend':'hugsim_geometry_sr_pbr_v1','source_report':str(REPORTS[scene]),'fps':10.,'fps_authority':'source_metadata_assumption','render_scale':1.,'camera_order':['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'],'vehicle_config':str(ROOT/'waymo23-native-import/configs/vehicle.json')}
   write(work/'scene_manifest.json',manifest);write(work/'map/map.json',map_data);write(work/'initial_context.json',initial)
   write(work/'scenario/scenario.json',{'config':{'intervention_start_s':0.,'initial_ego_speed_mps':speed},'edit_applied':False,'source_scene':scene})
   plan={'schema':'cross_scene_frozen_patch_episode_v1','scene':scene,'seed':template['seed'],'common_scene_token':f'heldout_scene{scene}_source{start}','initial_context':initial,'source_end_frame':len(poses)-1,'controls':controls,'scene_bundle':str(bundle),'candidate_code':candidate['code'],'candidate_pins':candidate['code_pins'],'mode':'off' if arm=='off' else 'on','navigation_command':COMMANDS[scene],'rigid_actor_count':meta['actors']['RigidNodes']['instances'],'actor_reference':str(OUT/'scene_actor_references'/str(scene)/'actor_tracks.npz'),'assets':{'uniad_weights':template['assets']['uniad_weights']},'native_collision_policy':template['native_collision_policy'],'termination':template['termination'],'reference_reuse':False,'fresh_history':True,'clock_assumption_note':'Existing 10Hz source assumption; hardware timestamps retained in raw metadata, no independent clock certification','full_original_scene_test':False,'map_authority':'reconstructed_support_proxy','physical_white_boundary_certified':False}
   verify(plan['assets']['uniad_weights']);write(work/'plan.json',plan)
   write(work/'input_pins.json',{str(p.resolve()):sha(p) for p in work.rglob('*') if p.is_file()})
   scene_row['arms'][arm]=str(work)
  cases[str(scene)]=scene_row
 m={'schema':'cross_scene_off_on_matrix_v1','best_measured':'v15','stable_comparison':'v17','cases':cases,'pins':source_pins(),'inputs':inputs,'selection_rule':'first source frame >=10 with conservative full-body stopping tube on observed support, 0.55m plus half-step velocity clearance and >=30 controls remaining; no outcome-based selection','score_formula_changed':False,'physics_changed':False,'controller_strategy_changed':False,'geometry_adapter_shared_by_all_arms':True,'full_scene_generalization_certified':False,'gpu_executed':False,'budgets':{'wall_seconds':14400,'max_episodes':6},'budget_scope':'New explicitly requested cross-scene validation only; old LLM campaign usage unchanged'}
 write(OUT/'matrix.json',m);return m

def initial_source_collision(plan,vehicle):
 """CPU audit of the actual source checkpoint; rendered heldout actors are rechecked live."""
 import torch
 import environment
 bundle=Path(plan['scene_bundle']);meta=read(bundle/'source_manifest.json');out=OUT/'cases'/str(plan['scene'])/'source_initial_collision.json'
 key={'scene':plan['scene'],'source_checkpoint_sha256':meta['source_checkpoint_sha256'],'source_frame':plan['initial_context']['source_frame'],'pose':plan['initial_context']['ego_to_world'],'point_semantics':asset(bundle/'point_semantics/background_semantics.npz')}
 if out.exists():
  r=read(out)
  if r['input_identity']!=key or not r['passed']:raise ValueError('Initial collision audit mismatch')
  return r
 state=torch.load(meta['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)['models'];opacity=torch.sigmoid(state['Background']['_opacities']).numpy()
 with np.load(bundle/'actors.npz') as z:tracks={k:z[k].copy() for k in z.files if k.split('__')[0] in ['RigidNodes','DeformableNodes','SMPLNodes']}
 for name in ['RigidNodes','DeformableNodes','SMPLNodes']:
  k=name+'__instances_quats'
  if k in tracks and tracks[k].ndim==4:tracks[k]=tracks[k][:,:,0,:]
 check=environment.Evaluation(tracks,vehicle,opacity,device='cpu').check(np.asarray(plan['initial_context']['ego_to_world']),plan['initial_context']['source_frame'])
 r={'passed':not check['collision'] and not check['native_route_deviation'] and not check['native_route_completed'],'input_identity':key,'native_source_geometry_check':check,'rendered_heldout_actor_check_still_required':True,'GPU_executed':False}
 write(out,r)
 if not r['passed']:raise ValueError('Initial source collision/route check failed before GPU launch')
 return r

def validate_scene_manifest(work,plan):
 from scene_api import SceneSpec
 spec=SceneSpec.read(Path(work)/'scene_manifest.json')
 if spec.source['scene']!=plan['scene'] or spec.frames!=plan['source_end_frame']+1:raise ValueError('Scene manifest identity/frame mismatch')
 if abs(spec.fps*spec.vehicle.dt-1.)>1e-9:raise ValueError('Scene/vehicle clocks disagree')
 if plan['source_end_frame']-plan['initial_context']['source_frame']!=plan['controls']:raise ValueError('Clock/control coverage mismatch')
 return spec

def cpu(work):
 common,runtime,scoring,diagnostics,changes=bind(work,read(Path(work)/'plan.json')['mode']);plan=read(Path(work)/'plan.json')
 spec=validate_scene_manifest(work,plan)
 from runtime_configuration import load_vehicle_config
 from build_source_sequence import training_lidar_mount
 from uniad_bridge import B,T
 import controller_switch
 v=load_vehicle_config();mount=training_lidar_mount(ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml');pose=np.asarray(plan['initial_context']['ego_to_world']);virtual=B@(T@pose@T.T)@B.T@mount
 info={'timestamp':0.,'scene_token':plan['common_scene_token'],'model_lidar_frame':{'virtual_to_ego':mount},'explicit_l2g_r_mat':virtual[:3,:3],'explicit_l2g_t':virtual[:3,3]}
 obj=controller_switch.JointReference(v,read(Path(work)/'map/map.json'));obj.observed({},info,{})
 states=np.zeros((31,5));states[:,:2]=-mount[:2,3];cert=obj.certificate(states)
 if not cert['passed'] or obj.geometry.pose_clearance(pose)<.05:raise ValueError('Registered initial pose fails actual geometry certificate: '+str(cert))
 roundtrip=obj.geometry.world(obj.geometry.model(np.array([pose[:2,3]])))
 if np.linalg.norm(roundtrip[0]-pose[:2,3])>.05:raise ValueError('Geometry coordinate roundtrip failed')
 if common.BUNDLE!=Path(plan['scene_bundle']) or scoring.BUNDLE!=common.BUNDLE:raise ValueError('Wrong scoring bundle')
 from live_env import Motion
 motion=Motion(v.dt,v.wheelbase_m,plan['initial_context']['initial_speed_mps'])
 stopping=obj.stop_states(motion,{'acc_mps2':v.min_acc_mps2,'steer_rate_radps':0.})
 stopping_certificate=obj.certificate(stopping)
 if not stopping_certificate['passed']:raise ValueError('Initial moving full-braking envelope not covered: '+str(stopping_certificate))
 initial_audit=initial_source_collision(plan,v)
 result={'source_initial_collision':asset(OUT/'cases'/str(plan['scene'])/'source_initial_collision.json'),'passed':True,'scene':plan['scene'],'source_frame':plan['initial_context']['source_frame'],'rigid_actor_count':plan['rigid_actor_count'],'mode':plan['mode'],'candidate_code':plan['candidate_code'],'scene_manifest_parsed_by_actual_runtime':True,'scene_manifest':asset(Path(work)/'scene_manifest.json'),'runtime_adaptations':changes,'initial_certificate':cert,'initial_stopping_certificate':stopping_certificate,'CPU_only':True,'GPU_executed':False}
 write(Path(work)/'cpu_admission.json',result);print(json.dumps(result,indent=2))

def execute(work):
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no experiment started')
 work=Path(work);plan=read(work/'plan.json');m=read(OUT/'matrix.json');verify_pins(m['pins'])
 for a in m['inputs']:verify(a)
 for p,h in read(work/'input_pins.json').items():
  if sha(p)!=h:raise ValueError('Episode input changed')
 if not read(work/'cpu_admission.json')['passed']:raise ValueError('CPU admission missing')
 common,runtime,scoring,diagnostics,changes=bind(work,plan['mode']);validate_scene_manifest(work,plan);started=time.time();r=runtime.run_arm(work,'A',plan['controls'],common=None,candidate=None)
 verify_pins(m['pins']);verify(plan['assets']['uniad_weights'])
 for a in m['inputs']:verify(a)
 r.update(scene=plan['scene'],model_weights_unchanged=True,experiment_plan=asset(work/'plan.json'),started_epoch=started,nav_command_source='preregistered task-level navigation; never source future ego trajectory',map_authority='reconstructed_support_proxy',truth_scope='observed reconstruction support only; not surveyed white boundary',comparison_mode=plan['mode'],full_original_scene_test=False,controls_selected_by='native_uniad_plus_shared_ilqr' if plan['mode']=='off' else 'frozen_external_patch_plus_shared_ilqr')
 write(work/'A/run_report.json',r)
 try:
  if len(r['trace'])>=25:scoring.score_arm(work,'A')
  else:write(work/'A/native_eval/native_scores.json',{'status':'insufficient_executed_horizon','scores':None,'executed_path_scored':False,'reason':'Fewer than 25 actual controls; no artificial future observations or actors'})
 except BaseException as e:write(work/'scoring_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
 summary=diagnostics.arm_summary(work/'A')
 write(work/'completion.json',{'report':asset(work/'A/run_report.json'),'summary':asset(work/'A/summary.json'),'plan':asset(work/'plan.json'),'pins':m['pins'],'status':r['status'],'scoring_error':read(work/'scoring_error.json') if (work/'scoring_error.json').exists() else None})
 if r['status'] in ('failed','resource_limited') or (work/'scoring_error.json').exists():raise RuntimeError('Episode infrastructure/scoring failure; remaining expensive episodes stopped: '+str(work))

def evaluate(m):
 result={'schema':'cross_scene_off_on_result_v1','cases':{},'global_patch_validated':False,'full_original_scene_test':False,'score_comparison':'native actual executed-path scores; never predicted trajectories'}
 for s,c in m['cases'].items():
  if not c['admitted']:result['cases'][s]=c;continue
  arms={}
  for name,p in c['arms'].items():
   done=Path(p)/'completion.json'
   if not done.exists():
    failure=Path(p)/'launch_failure.json';arms[name]={'status':'infrastructure_failed' if failure.exists() else 'pending','failure':read(failure) if failure.exists() else None};continue
   r=read(done)
   for key in ['report','summary','plan']:verify(r[key])
   summary=read(r['summary']['path']);scores=(summary.get('native_scores') or {}).get('scores') or {}
   arms[name]={'status':summary['status'],'controls':summary['controls'],'observations':summary['observations'],'termination_reason':summary['termination_reason'],'any_anomaly':summary['any_anomaly'],'stop_unresolved':summary['stop_recovery_unresolved'],'scores':scores.get('executed_path'),'completion':asset(done)}
  comparisons={}
  for name in ['v15','v17']:
   b,n=arms['off'],arms[name];comparable=all(x.get('status')=='completed' and x.get('controls')==c['controls'] and x.get('scores') for x in [b,n])
   deltas={k:n['scores'][k]['value']-b['scores'][k]['value'] for k in ['hdscore','pdms','rc']} if comparable else None
   comparisons[name]={'comparable_full_segment':bool(comparable),'deltas':deltas,'segment_nonregression':bool(comparable and not n['any_anomaly'] and not n['stop_unresolved'] and all(x>=-1e-9 for x in deltas.values()))}
  result['cases'][s]={'scope':c['scope'],'source_start':c['start_frame'],'planned_controls':c['controls'],'arms':arms,'comparisons':comparisons}
 write(OUT/'evaluation.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','check','full','status','_cpu','_execute']);parser.add_argument('--work',type=Path);a=parser.parse_args()
 if a.stage=='_cpu':cpu(a.work);return
 if a.stage=='_execute':
  try:execute(a.work)
  except BaseException as e:
   report_path=a.work/'A/run_report.json';r=read(report_path) if report_path.exists() else {}
   write(a.work/'launch_failure.json',{'classification':'execution_or_infrastructure_failure','error':repr(e),'traceback':traceback.format_exc(),'actual_controls':len(r.get('trace',[])),'actual_observations':len(r.get('observations',[])),'not_a_physical_patch_failure_without_executed_evidence':not bool(r.get('trace'))})
   raise
  return
 if a.stage=='status':print(json.dumps(read(OUT/'evaluation.json') if (OUT/'evaluation.json').exists() else {'status':'not_executed'},indent=2));return
 m=prepare()
 if a.stage in ['check','full']:
  preflight=OUT/'smpl_preflight.json'
  subprocess.run([sys.executable,'-u','-B',str(ROOT/'integrations/autonomous-patch-agent-v8/probe.py'),'--code',str(CANON),'--output',str(preflight)],check=True)
  if not read(preflight)['passed']:raise ValueError('Actual neutral SMPL CPU constructor failed')
  for c in m['cases'].values():
   if c['admitted']:
    for work in c['arms'].values():subprocess.run([sys.executable,'-u','-B',str(LOCAL/'pipeline.py'),'_cpu','--work',work],check=True)
  write(OUT/'cpu_admission.json',{'passed':True,'registered_cases':list(m['cases']),'runnable_cases':[s for s,c in m['cases'].items() if c['admitted']],'blocked_cases':[s for s,c in m['cases'].items() if not c['admitted']],'pins':m['pins'],'GPU_executed':False})
 if a.stage=='full':
  import torch
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU admission saved but no GPU run started')
  with (P26/'cross_scene_validation_v1/coordinator.lock').open('a') as coordinator:
   fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB)
   started_path=OUT/'started.json'
   if not started_path.exists():write(started_path,{'epoch':time.time()})
   deadline=read(started_path)['epoch']+m['budgets']['wall_seconds']
   with (ROOT/'outputs/gpu_test_service/worker.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('GPU lock busy; no episodes started')
    for scene,c in m['cases'].items():
     if not c['admitted']:continue
     for arm,work in c['arms'].items():
      if (OUT/'STOP').exists() or (ROOT/'outputs/gpu_test_service/STOP').exists():evaluate(m);return
      if (Path(work)/'completion.json').exists():
       completed=read(Path(work)/'completion.json')
       if completed['status'] in ('failed','resource_limited') or completed.get('scoring_error'):raise RuntimeError('Previous failed episode retained; not reusable as success: '+work)
       continue
      if (Path(work)/'A').exists():raise RuntimeError('Partial episode retained; no replay or hidden overwrite: '+work)
      remaining=deadline-time.time()
      if remaining<=0:raise RuntimeError('Shared cross-scene wall budget exhausted')
      write(OUT/'progress.json',{'scene':scene,'arm':arm,'stage':'fresh_closed_loop','wall_remaining_s':remaining})
      try:subprocess.run([sys.executable,'-u','-B',str(LOCAL/'pipeline.py'),'_execute','--work',work],check=True,timeout=remaining)
      except BaseException:evaluate(m);raise
      evaluate(m)
 result=evaluate(m);print(json.dumps({'output':str(OUT),'runnable_cases':[s for s,c in m['cases'].items() if c['admitted']],'blocked_cases':[s for s,c in m['cases'].items() if not c['admitted']],'global_patch_validated':False},indent=2))
if __name__=='__main__':main()
