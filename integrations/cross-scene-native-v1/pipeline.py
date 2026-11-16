"""Offline scene admission and fresh original-scene native comparison jobs."""
import argparse,copy,fcntl,os,shutil,subprocess,sys,time
import numpy as np
from scene_contracts import *
from projection_domain import REVISION,POINT_FOLDER
from scoring_assets import ensure_view
PYTHON=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'

def initial_context(scene,vehicle):
 bundle=ASSETS/str(scene);poses=np.load(bundle/'poses_and_clock.npz')['relative_ego_to_sr_world'];frame=10
 speed=float(np.linalg.norm(poses[frame,:2,3]-poses[frame-1,:2,3])/.1)
 if not 0<=speed<=vehicle.max_speed_mps:raise ValueError('Recorded initial speed outside physical vehicle range')
 return {'source_frame':frame,'source_timestamp_s':frame*.1,'local_timestamp_start_s':0.,'ego_to_world':poses[frame].tolist(),'initial_speed_mps':speed,'initial_steering_rad':0.,'initial_speed_source':'current/backward recorded pose difference; identical for comparison arms','initial_pose_set_once_only':True,'history':'fresh full UniAD history; no saved tensors or actions reused'}

def source_admission(scene,mapping):
 source=read(ASSETS/str(scene)/'source_manifest.json');report=read(REPORTS[scene]);raw=read(ASSETS/str(scene)/'raw_metadata/raw_metadata.json')
 if source['scene']!=scene or report['frames']!=source['frames'] or raw['scene_index']!=scene:raise ValueError('Scene/source identity mismatch')
 sys.path.insert(0,str(ROOT/'waymo23-native-import'));from runtime_configuration import load_vehicle_config
 vehicle=load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json');ctx=initial_context(scene,vehicle)
 from shapely.geometry import shape,Polygon
 poly=shape(mapping['geometry']);pose=np.asarray(ctx['ego_to_world']);corners=np.array([[-vehicle.half_length_m,-vehicle.half_width_m,0],[vehicle.half_length_m,-vehicle.half_width_m,0],[vehicle.half_length_m,vehicle.half_width_m,0],[-vehicle.half_length_m,vehicle.half_width_m,0]])
 body=Polygon((corners@pose[:3,:3].T+pose[:3,3])[:,:2]);gap=body.distance(poly.boundary)-vehicle.clearance_margin_m-mapping['uncertainty_m'] if poly.covers(body) else -1.
 return {'scene':scene,'passed':bool(gap>=.05),'initial_footprint_gap_m':float(gap),'initial_context':ctx,'source_manifest':asset(ASSETS/str(scene)/'source_manifest.json'),'map':asset(MAPS/str(scene)/'map.json'),'source_HD_map_available':False,'boundary_is_sensor_evidence_only':True,'required_followup':'GPU renderer/model boot and fresh independent closed loops','GPU_executed':False}

def prepare(scene,version,controls=100):
 row=policy(version);folder=OUT/'private_validation/episodes'/str(scene)/version;bundle=ASSETS/str(scene);mapping=read(MAPS/str(scene)/'map.json');admission=source_admission(scene,mapping)
 if not admission['passed']:raise ValueError('Scene road support does not admit initial full body; inspect scene_admission.json')
 scoring_bundle=ensure_view(scene)
 if (folder/'inputs.json').exists():
  previous=read(folder/'plan.json');frozen=read(folder/'policy.json')
  if previous['scene']!=scene or previous['controls']!=controls or frozen['code_pins']!=row['code_pins']:raise ValueError('Cached episode belongs to another frozen problem')
  verify_pins(read(folder/'inputs.json'));return folder
 if folder.exists():raise ValueError('Partial preparation retained: '+str(folder))
 folder.mkdir(parents=True);(folder/'map').mkdir();(folder/'scenario').mkdir();(folder/'boot').mkdir()
 release=Path(row['release']);template=read(release/'plan.json');plan=copy.deepcopy(template)
 for name in ('manifest.json','legacy_support.zip'):
  if (release/'boot'/name).exists():shutil.copyfile(release/'boot'/name,folder/'boot'/name)
 shutil.copyfile(MAPS/str(scene)/'map.json',folder/'map/map.json');shutil.copyfile(bundle/'actors.npz',folder/'source_actor_tracks.npz')
 write(folder/'source_metadata.json',read(bundle/'source_manifest.json'));write(folder/'policy.json',row);write(folder/'scene_verification.json',admission)
 write(folder/'baseline_freeze.json',{'verification':asset(folder/'scene_verification.json'),'controller':row['code_pins'],'new_history_required':True,'no_551_actor_data_reused':True})
 write(folder/'scene_manifest.json',{'schema':'unified_scene_v1','backend':'hugsim_geometry_sr_pbr_v1','source_report':str(REPORTS[scene]),'fps':10.,'fps_authority':'source_metadata_assumption','render_scale':1.,'camera_order':['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'],'vehicle_config':str(ROOT/'waymo23-native-import/configs/vehicle.json')})
 ctx=admission['initial_context'];frames=read(bundle/'source_manifest.json')['frames']
 if type(controls) is not int or controls<25 or ctx['source_frame']+controls>=frames:raise ValueError('Horizon outside actual scene support')
 write(folder/'scenario/scenario.json',{'config':{'initial_ego_speed_mps':ctx['initial_speed_mps'],'intervention_start_s':0.,'absolute_source_activation_s':0.},'edit':None,'all_original_native_actors':True})
 # Replace every scene-dependent asset; old score/controller inputs never travel.
 plan.update(scene=scene,navigation_command=COMMANDS[scene],controls=controls,source_end_frame=frames-1,initial_context=ctx,common_scene_token='external_scene_'+str(scene),baseline_root=str(folder),prefix_policy='none; independently inferred full histories',old_snapshots_reused=False,baseline_reused_without_reexecution=False,gpu_executed=False)
 plan['assets']={'uniad_weights':template['assets']['uniad_weights'],'scene':asset(folder/'scene_manifest.json'),'source_manifest':asset(bundle/'source_manifest.json'),'scenario':asset(folder/'scenario/scenario.json'),'baseline':asset(folder/'baseline_freeze.json')}
 for key in ('previous_A','previous_plan','previous_B','baseline_receipts','old_A_v3','collision_off_baseline_root','collision_off_baseline_report','preregistration','native_cpu_admission','initial_context_asset','static_map','boot_manifest','boot_archive','new_A_required'):
  plan.pop(key,None)
 plan['static_map']=asset(folder/'map/map.json');plan['boot_manifest']=asset(folder/'boot/manifest.json');plan['allowed_online_inputs']=template['allowed_online_inputs'];plan['forbidden_online_inputs']=template['forbidden_online_inputs']
 write(folder/'plan.json',plan)
 inputs={str(p):sha(p) for p in folder.rglob('*') if p.is_file()};inputs.update(row['code_pins']);inputs.update(pins());inputs.update(template['code']);inputs.update(template['runtime_code']);inputs.update(template['map_v4_code'])
 verify(plan['assets']['uniad_weights']);inputs[plan['assets']['uniad_weights']['path']]=plan['assets']['uniad_weights']['sha256']
 # Include every map-support source, not only the generated polygon.
 for entry in mapping['sources']+[mapping['support']]:verify(entry);inputs[entry['path']]=entry['sha256']
 for name in ('source_manifest.json','point_semantics/point_semantics_manifest.json','point_semantics/background_semantics.npz','actors.npz','native_dense_route.npy','background_sr_world.npy','road_sr_world.npy'):inputs[str(scoring_bundle/name)]=sha(scoring_bundle/name)
 inputs[str(REPORTS[scene])]=sha(REPORTS[scene]);write(folder/'inputs.json',inputs);return folder

def execute(scene,version,controls):
 cpu=read(OUT/'cpu_admission.json');pool=read(OUT/'validation_pool.json')
 if not cpu['passed'] or cpu['tests']<18 or cpu['code_pins']!=pins():raise RuntimeError('Current host CPU admission required')
 if scene not in pool['scene_ids'] or controls!=pool['controls']:raise ValueError('Episode outside frozen asset-admitted validation pool')
 if policy(version)['code_pins']!=pool['policies'][version]['code_pins']:raise ValueError('Frozen comparison controller changed')
 verify(pool['admission_snapshots'][str(scene)]['road_map'])
 os.fstat(9)
 for key in list(os.environ):
  if key.startswith(('PATCH_','OPENAI_','ANTHROPIC_')) or key in ('API_KEY','HF_TOKEN'):os.environ.pop(key,None)
 folder=prepare(scene,version,controls);verify_pins(read(folder/'inputs.json'))
 if (folder/'completion.json').exists():
  done=read(folder/'completion.json')
  for value in done['outputs'].values():verify(value)
  return done
 if (folder/'A').exists():raise ValueError('Partial independent episode retained; never stitch old controls')
 from runtime_adapter import load
 row=read(folder/'policy.json');runtime,scoring,audit=load(row,ensure_view(scene));write(folder/'runtime_binding.json',audit)
 from budget import reserve
 with reserve(folder) as remaining:
  import signal
  previous=signal.getsignal(signal.SIGALRM);began=time.monotonic()
  def expired(*args):raise TimeoutError('Shared cumulative episode wall allowance exhausted')
  signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
  try:
   report=runtime.run_arm(folder,'A',controls,common=None,candidate=None)
   report.update(experiment_plan=asset(folder/'plan.json'),model_weights_unchanged=None,nav_command_source='pre-registered per-scene command',cross_scene_binding=asset(folder/'runtime_binding.json'),original_scene_only=True)
   write(folder/'A/run_report.json',report)
   remaining_score=remaining-(time.monotonic()-began)
   if remaining_score<=0:raise TimeoutError('Shared wall allowance exhausted before native scoring')
   # Only a resource deadline changes; scoring formula/inputs remain frozen.
   runtime.TERMINATION['native_score_wall_limit_s']=min(runtime.TERMINATION['native_score_wall_limit_s'],remaining_score)
   scoring.score_arm(folder,'A')
   verify_pins(read(folder/'inputs.json'));report['model_weights_unchanged']=True;write(folder/'A/run_report.json',report)
   import diagnostics
   diagnostics.arm_summary(folder/'A')
   if time.monotonic()-began>remaining:raise TimeoutError('Shared wall allowance exhausted; no completed admission')
  finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
  done={'status':report['status'],'scene':scene,'version':version,'controls':len(report['trace']),'termination_reason':report.get('termination_reason'),'outputs':{'report':asset(folder/'A/run_report.json'),'summary':asset(folder/'A/summary.json'),'scores':asset(folder/'A/native_eval/native_scores.json')},'global_patch_validated':False}
  write(folder/'completion.json',done);return done

def status():
 rows={}
 for s in REPORTS:
  a=MAPS/str(s)/'scene_admission.json';progress=MAPS/str(s)/'progress.json'
  rows[str(s)]={'map_admission':read(a) if a.exists() else None,'map_progress':read(progress) if progress.exists() else None,'episodes':{v:{'finished':True,'details_withheld_until_host_gate':True} if (OUT/'private_validation/episodes'/str(s)/v/'completion.json').exists() else None for v in ('v13','v15','v17')}}
 return {'scenes':rows,'global_patch_validated':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['maps','cpu','data','bootstrap','extend-window','asset-stage','prepare','check','bind','execute','full','status']);p.add_argument('--scene',type=int,choices=REPORTS);p.add_argument('--version',choices=['v13','v15','v17'],default='v15');p.add_argument('--controls',type=int,default=100);p.add_argument('--seconds',type=int);p.add_argument('--asset-kind',choices=['data','raw','semantics','points']);a=p.parse_args();scenes=[a.scene] if a.scene else list(REPORTS);OUT.mkdir(parents=True,exist_ok=True)
 if a.stage=='status':print(__import__('json').dumps(status(),ensure_ascii=False,indent=2));return
 if a.stage=='cpu':
  import re
  result=subprocess.run([str(PYTHON),'-B','-m','unittest','discover','-s',str(LOCAL/'tests'),'-v'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  print(result.stdout);count=re.search(r'Ran (\d+) tests',result.stdout);write(OUT/'cpu_admission.json',{'passed':result.returncode==0,'tests':int(count[1]) if count else 0,'code_pins':pins(),'GPU_executed':False});raise SystemExit(result.returncode)
 elif a.stage=='data':
  from extra_assets import stage,EXTRA
  for scene in EXTRA:
   for kind in ('data','raw'):stage(scene,kind)
 elif a.stage=='extend-window':
  from budget import extend_window
  print(__import__('json').dumps(extend_window(a.seconds)))
 elif a.stage=='bootstrap':
  from extra_assets import stage,EXTRA
  from budget import extend_window,reserve,readiness
  os.fstat(9)
  if a.seconds is not None:extend_window(a.seconds)
  if not readiness()['ready']:raise RuntimeError('Shared wall budget expired; use explicit --seconds for an additive window')
  for scene in EXTRA:
   for kind in ('data','raw'):stage(scene,kind)
   for kind,manifest in (('semantics','semantic19/semantic_manifest.json'),('points',POINT_FOLDER+'/point_semantics_manifest.json')):
    if kind=='semantics' and not (ASSETS/str(scene)/manifest).exists():
     with reserve(OUT/'asset_jobs'/str(scene)/kind):stage(scene,kind)
    else:stage(scene,kind)
   from boundaries import build
   data=build(scene);write(MAPS/str(scene)/'scene_admission.json',source_admission(scene,data))
  subprocess.run(['bash',str(LOCAL/'run.sh'),'check'],check=True)
 elif a.stage=='asset-stage':
  if a.scene is None or a.asset_kind is None:raise ValueError('Asset stage requires scene and kind')
  from extra_assets import stage
  if a.asset_kind=='semantics' and not (ASSETS/str(a.scene)/'semantic19/semantic_manifest.json').exists():
   os.fstat(9)
   from budget import reserve
   with reserve(OUT/'asset_jobs'/str(a.scene)/a.asset_kind):stage(a.scene,a.asset_kind)
  else:stage(a.scene,a.asset_kind)
 elif a.stage=='maps':
  from boundaries import build
  for scene in scenes:
   mapping=build(scene);write(MAPS/str(scene)/'scene_admission.json',source_admission(scene,mapping))
 elif a.stage=='prepare':
  for scene in scenes:prepare(scene,a.version,a.controls)
 elif a.stage=='bind':
  if a.scene is None:raise ValueError('Bind requires exactly one scene')
  from runtime_adapter import load
  runtime,scoring,audit=load(policy(a.version),NATIVE_ASSETS/str(a.scene));write(OUT/'bindings'/str(a.scene)/(a.version+'.json'),dict(audit,scene=a.scene,source_bundle=str(scoring.BUNDLE),CPU_binding_passed=True,GPU_executed=False))
 elif a.stage=='check':
  report={'schema':'cross_scene_runtime_admission_v1','scenes':{},'GPU_executed':False,'global_patch_validated':False}
  for scene in scenes:
   file=MAPS/str(scene)/'map.json'
   if not file.exists():report['scenes'][str(scene)]={'admitted':False,'missing':['scene-specific road domain']};continue
   data=read(file)
   for entry in data['sources']+[data['support']]:verify(entry)
   admission=source_admission(scene,data);bindings={}
   for version in ('v13','v15','v17'):
    subprocess.run([str(PYTHON),'-B',str(LOCAL/'pipeline.py'),'bind','--scene',str(scene),'--version',version],check=True)
    bindings[version]=asset(OUT/'bindings'/str(scene)/(version+'.json'))
   report['scenes'][str(scene)]={'admitted':admission['passed'],'initial_footprint_gap_m':admission['initial_footprint_gap_m'],'road_map':admission['map'],'runtime_bindings':bindings,'missing':([] if admission['passed'] else ['Original start full-body road coverage; unknown cannot be made free'])+([] if (ASSETS/str(scene)/POINT_FOLDER/'point_semantics_manifest.json').exists() else ['Corrected point semantics for native scoring'])}
  from budget import readiness
  report['shared_budget']=readiness()
  eligible=[int(scene) for scene,value in report['scenes'].items() if value['admitted'] and not value['missing']]
  poolfile=OUT/'validation_pool.json'
  if len(eligible)>=2 and not poolfile.exists():
   write(poolfile,{'schema':'exogenous_asset_admitted_validation_pool_v1','scene_ids':sorted(eligible),'admission_snapshots':{str(scene):report['scenes'][str(scene)] for scene in eligible},'policies':{version:policy(version) for version in ('v13','v15','v17')},'selected_before_any_controller_scores':True,'excluded_only_for_asset_coverage':{scene:value['missing'] for scene,value in report['scenes'].items() if not value['admitted']},'controls':a.controls,'source_start_frame':10})
  report['ready_for_GPU']=poolfile.exists() and report['shared_budget']['ready'];report['frozen_pool']=asset(poolfile) if poolfile.exists() else None;write(OUT/'admission_report.json',report)
  print(__import__('json').dumps(report,ensure_ascii=False,indent=2))
 elif a.stage=='execute':
  if a.scene is None:raise ValueError('Execute requires one scene')
  private=OUT/'private_validation';private.mkdir(parents=True,exist_ok=True);private.chmod(0o700)
  import contextlib
  with (private/(str(a.scene)+'_'+a.version+'.log')).open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):execute(a.scene,a.version,a.controls)
  print('Native episode finished; detailed results remain private pending host gate')
 elif a.stage=='full':
  poolfile=OUT/'validation_pool.json'
  if not poolfile.exists():raise RuntimeError('No two asset-admitted held-out scenes; no GPU episodes started')
  pool=read(poolfile)
  for frozen in pool['admission_snapshots'].values():
   verify(frozen['road_map'])
   for binding in frozen['runtime_bindings'].values():verify(binding)
  scenes=[a.scene] if a.scene is not None else pool['scene_ids']
  if any(scene not in pool['scene_ids'] for scene in scenes) or a.controls!=pool['controls']:raise ValueError('Frozen external pool/horizon changed')
  cpu=read(OUT/'cpu_admission.json')
  if not cpu['passed'] or cpu['tests']<18 or cpu['code_pins']!=pins():raise RuntimeError('Current host CPU admission required before GPU matrix')
  # Source/runtime checks must finish for every arm before any expensive GPU work.
  for scene in scenes:
   for version in ('v13','v15','v17'):prepare(scene,version,a.controls)
  for scene in scenes:
   for version in ('v13','v15','v17'):subprocess.run(['bash',str(LOCAL/'run.sh'),'execute','--scene',str(scene),'--version',version,'--controls',str(a.controls)],check=True)
 if a.stage not in ('execute','full'):print('Stage complete:',a.stage)
if __name__=='__main__':main()
