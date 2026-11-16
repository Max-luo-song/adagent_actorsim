"""Fresh paired validation. No old actions, hidden overwrite, model changes or promotion."""

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

import argparse,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,signal
pass  # Path is provided by the portable boundary above
import numpy as np
from bindings import ROOT,LOCAL,PARENT,REPAIR,CANON,MAP,OLD,CROSS,bind
from gates import compare,replay_gate,values
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26';OUT=P26/'box_validity_validation_v2';RECORD=P26/'box_validity_repair_v1'
ARMS=('off','v15','repair');CASES={'551_original':(P26/'patch_task_discovery_v1/pair/episodes/original_on','A'),'551_cutin':(P26/'patch_task_discovery_v1/pair/episodes/edited_on','B'),'172':(P26/'cross_scene_validation_v2/cases/172/v15','A'),'552':(P26/'cross_scene_validation_v2/cases/552/v15','A')}
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Changed asset: '+a['path'])
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def pins():
 inherited=read(P26/'cross_scene_validation_v2/matrix.json')['pins']
 for p,h in inherited.items():
  if sha(p)!=h:raise ValueError('Inherited frozen release changed: '+p)
 paths=[*LOCAL.rglob('*.py'),LOCAL/'run.sh',*REPAIR.rglob('*.py'),*PARENT.rglob('*.py')]
 return inherited|{str(p.resolve()):sha(p) for p in paths}
def verify_pins(p):
 for name,h in p.items():
  if sha(name)!=h:raise ValueError('Validation release changed: '+name)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'matrix.json').exists():
  m=read(OUT/'matrix.json');verify_pins(m['pins'])
  for a in m['inputs']:verify(a)
  return m
 previous=P26/'box_validity_validation_v1';old=read(previous/'matrix.json');verify_pins(old['pins'])
 for a in old['inputs']:verify(a)
 cases=_hugsim_expand(json.loads(json.dumps(old['cases'])));inputs=list(old['inputs'])+[asset(previous/'matrix.json')];overrides={};reused=0;endpoints=[]
 for case,c in cases.items():
  for arm,w in list(c['arms'].items()):
   work=Path(w);done=work/'completion.json'
   if done.exists():
    receipt=read(done)
    for k in ['report','summary','plan']:verify(receipt[k])
    inputs.extend([asset(done),receipt['report'],receipt['summary'],receipt['plan']]);endpoints.append(done.stat().st_mtime);reused+=1
    if receipt.get('scoring_error'):
     # Repair postprocessing only; do not replay or mutate any recorded controls.
     recovery=OUT/'recovered'/case/arm;recovery.mkdir(parents=True)
     report=read(receipt['report']['path'])
     from short_history import insufficient_result
     result=insufficient_result(report)
     if result is None:raise ValueError('Unregistered scoring failure requires investigation: '+w)
     write(recovery/'native_scores.json',result);summary=read(receipt['summary']['path']);summary['native_scores']=result
     write(recovery/'summary.json',summary)
     fixed=dict(receipt,summary=asset(recovery/'summary.json'),scoring_error=False,postprocessing_only=True,original_failure=asset(work/'scoring_error.json'),new_controls_executed=0)
     write(recovery/'completion.json',fixed);overrides[w]=str(recovery/'completion.json');inputs.extend([asset(recovery/'completion.json'),asset(recovery/'native_scores.json'),fixed['summary'],fixed['original_failure']])
    continue
   if (work/c['episode_name']).exists():raise ValueError('Partial live episode retained; no stitching: '+w)
   new=OUT/'cases'/case/arm;new.mkdir(parents=True)
   for rel in ['scene_manifest.json','baseline_freeze.json','baseline_verification.json','initial_context.json','map/map.json','boot/manifest.json','scenario/scenario.json','scenario/target_track.npz']:
    f=work/rel
    if not f.exists():continue
    target=new/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
   plan=read(work/'plan.json');write(new/'plan.json',plan)
   write(new/'input_pins.json',{str(f.resolve()):sha(f) for f in new.rglob('*') if f.is_file()});write(new/'input_receipt.json',asset(new/'input_pins.json'));inputs.append(asset(new/'input_receipt.json'));c['arms'][arm]=str(new)
 started=read(previous/'started.json')['epoch'];spent=max(endpoints)-started
 if not 0<=spent<old['budget']['wall_seconds']:raise ValueError('Previous validation budget exhausted or clock invalid')
 m=dict(old,schema='box_validity_validation_continuation_v2',cases=cases,pins=pins(),inputs=list({a['path']:a for a in inputs}.values()),postprocess_overrides=overrides,completed_episodes_reused=reused,old_results_reused=True,previous_matrix=asset(previous/'matrix.json'),budget=dict(old['budget'],previous_elapsed_seconds=spent,wall_seconds=old['budget']['wall_seconds']-spent,original_total_wall_seconds=old['budget']['wall_seconds'],scope='Carry remaining active-run budget; downtime after retained scoring failure excluded and recorded, no old LLM budget reset'),user_policy='Lateral offsets are diagnostics only, not an acceptance veto; compare native metrics, original-scene nonregression and actual safety')
 write(OUT/'matrix.json',m);return m

def cpu(work):
 work=Path(work);plan=read(work/'plan.json');common,runtime,scoring,diagnostics,changes=bind(work)
 from scene_api import SceneSpec
 from build_source_sequence import training_lidar_mount
 from uniad_bridge import B,T
 import controller_switch
 spec=SceneSpec.read(work/'scene_manifest.json');v=spec.vehicle
 verify(read(work/'baseline_freeze.json')['verification'])
 if spec.source['scene']!=plan['scene'] or spec.frames!=plan['source_end_frame']+1 or abs(v.dt*spec.fps-1)>1e-9:raise ValueError('Scene/clock identity mismatch')
 if plan['initial_context']['source_frame']+plan['controls']>plan['source_end_frame']:raise ValueError('Insufficient source horizon')
 mount=training_lidar_mount(ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml');pose=np.asarray(plan['initial_context']['ego_to_world']);virtual=B@(T@pose@T.T)@B.T@mount
 info={'timestamp':0.,'scene_token':plan['common_scene_token'],'model_lidar_frame':{'virtual_to_ego':mount},'explicit_l2g_r_mat':virtual[:3,:3],'explicit_l2g_t':virtual[:3,3]}
 obj=controller_switch.JointReference(v,read(work/'map/map.json'));obj.observed({},info,{})
 from live_env import Motion
 motion=Motion(v.dt,v.wheelbase_m,plan['initial_context']['initial_speed_mps']);states=obj.stop_states(motion,{'acc_mps2':v.min_acc_mps2,'steer_rate_radps':0.});cert=obj.certificate(states)
 if not cert['passed'] or obj.geometry.pose_clearance(pose)<.05:raise ValueError('Initial full-body stopping geometry invalid')
 if common.BUNDLE!=scoring.BUNDLE:raise ValueError('Wrong native scoring scene bundle')
 if plan['validation_arm']=='repair':
  if not obj.supports_box_observation_contract or not hasattr(runtime,'attach_box_observation'):raise ValueError('Repair not connected to actual runtime')
  obj.require_box_observation_contract=True
  try:obj.update_tracks({},info)
  except ValueError:pass
  else:raise ValueError('Live repair accepts missing contract')
 write(work/'cpu_admission.json',{'passed':True,'source_scene':spec.source['scene'],'controller_file':str(sys.modules['joint_safety'].__file__),'runtime_scene_adaptations':changes,'initial_stopping_certificate':cert,'repair_attached':plan['validation_arm']=='repair','GPU_executed':False})

def episode_result(work):
 work=Path(work);p=read(work/'plan.json');overrides=read(OUT/'matrix.json').get('postprocess_overrides',{});f=Path(overrides.get(str(work),str(work/'completion.json')))
 if not f.exists():return {'status':'pending','progress':read(work/p['episode_name']/'progress.json') if (work/p['episode_name']/'progress.json').exists() else None}
 done=read(f)
 for k in ['report','summary','plan']:verify(done[k])
 r=read(done['report']['path']);s=read(done['summary']['path'])
 positions=np.asarray([o['pose'] for o in r['observations']])[:,:2,3] if r['observations'] else np.empty((0,2))
 anchor=np.asarray(p['initial_context']['ego_to_world']);lateral=np.abs((positions-anchor[:2,3])@anchor[:2,1]) if len(positions) else np.array([float('inf')])
 native=s.get('native_scores') or {};scores=(native.get('scores') or {}).get('executed_path')
 return {'status':r['status'],'controls':len(r['trace']),'scores':scores,'fresh_history':r.get('validation_fresh_history',False),'executed_path_scored':native.get('executed_path_scored',False),'safe':not s.get('any_anomaly',True) and not s.get('stop_recovery_unresolved',True),'termination_reason':r.get('termination_reason'),'final_abs_lateral_m':float(lateral[-1]),'peak_abs_lateral_m':float(lateral.max()),'completion':asset(f)}
def evaluate(m):
 cases={}
 for case,c in m['cases'].items():
  arms={a:episode_result(w) for a,w in c['arms'].items()};comparisons={a:compare(arms[a],arms['repair'],c['controls']) for a in ['off','v15']}
  protected=None;lateral_check=None
  if case.startswith('551_'):
   minimum=m['protected_original_hdscore'] if case=='551_original' else m['protected_target_hdscore']
   v=values(arms['repair'].get('scores'));protected=bool(v and v['hdscore']>=minimum-1e-9)
   lateral_check=all(isinstance(arms[a].get(k),(int,float)) for a in ['v15','repair'] for k in ['final_abs_lateral_m','peak_abs_lateral_m']) and all(arms['repair'][k]<=arms['v15'][k]+1e-6 for k in ['final_abs_lateral_m','peak_abs_lateral_m'])
  cases[case]={'arms':arms,'comparisons':comparisons,'protected_best_score_preserved':protected,'lateral_side_effect_not_worsened_vs_parent':lateral_check,'lateral_is_acceptance_gate':False,'passed':comparisons['v15']['passed'] and protected is not False}
 tested=all(c['passed'] for c in cases.values())
 r={'cases':cases,'tested_episode_nonregression_passed':tested,'global_patch_validated':False,'automatic_promotion':False,'reason':'No automatic resident promotion; blocked registered scene621 remains and reconstruction-support segments are not full-scene certification','unavailable_scenes':m['required_unavailable_scenes']}
 write(OUT/'evaluation.json',r);return r

def execute(work):
 lease=int(os.environ.get('BOX_VALIDATION_GPU_LEASE_FD','-1'))
 if lease<0:raise ValueError('GPU execution requires full-stage shared lock lease')
 if Path(os.readlink('/proc/self/fd/'+str(lease))).resolve()!=ROOT/'outputs/gpu_test_service/worker.lock':raise ValueError('Wrong GPU lease')
 work=Path(work);m=read(OUT/'matrix.json');verify_pins(m['pins'])
 for a in m['inputs']:verify(a)
 verify(read(work/'input_receipt.json'))
 for p,h in read(work/'input_pins.json').items():
  if sha(p)!=h:raise ValueError('Episode inputs changed: '+p)
 verify(read(OUT/'cpu_admission.json')['release'])
 if not read(work/'cpu_admission.json')['passed']:raise ValueError('CPU admission missing')
 p=read(work/'plan.json');common,runtime,scoring,diagnostics,changes=bind(work)
 r=runtime.run_arm(work,p['episode_name'],p['controls'],common=None,candidate=None)
 verify_pins(m['pins']);verify(p['assets']['uniad_weights'])
 for a in m['inputs']:verify(a)
 verify(read(work/'input_receipt.json'))
 for name,h in read(work/'input_pins.json').items():
  if sha(name)!=h:raise ValueError('Frozen episode input changed during execution: '+name)
 r.update(validation_fresh_history=True,validation_arm=p['validation_arm'],validation_case=p['validation_case'],experiment_plan=asset(work/'plan.json'),model_weights_unchanged=True)
 write(work/p['episode_name']/'run_report.json',r)
 try:
  from short_history import score
  score(scoring,work,p['episode_name'],write)
 except BaseException as e:write(work/'scoring_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
 diagnostics.arm_summary(work/p['episode_name'])
 write(work/'completion.json',{'report':asset(work/p['episode_name']/'run_report.json'),'summary':asset(work/p['episode_name']/'summary.json'),'plan':asset(work/'plan.json'),'status':r['status'],'scoring_error':(work/'scoring_error.json').exists()})
 if r['status'] in ['failed','resource_limited'] or (work/'scoring_error.json').exists():raise RuntimeError('Infrastructure/scoring failure retained; no automatic rerun')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','check','full','status','brief','_cpu','_execute']);parser.add_argument('--work',type=Path);a=parser.parse_args()
 if a.stage=='_cpu':cpu(a.work);return
 if a.stage=='_execute':
  try:execute(a.work)
  except BaseException as e:write(a.work/'launch_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'classification':'execution_error_not_assumed_physical_failure'});raise
  return
 if a.stage in ['status','brief']:
  if not (OUT/'matrix.json').exists():print('{"status":"not_prepared"}');return
  m=read(OUT/'matrix.json');r=evaluate(m)
  if a.stage=='brief':
   progress=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'stage':'prepared_or_CPU_preflight'}
   print('Stage:',progress.get('stage'),'Case:',progress.get('case','-'),'Arm:',progress.get('arm','-'))
   for k,c in r['cases'].items():print(k, ' | '.join(a+':'+v['status']+('/'+str(v['controls']) if 'controls' in v else '') for a,v in c['arms'].items()))
   print('Global validation: pending; blocked scene621; no resident promotion');return
  print(json.dumps({'output':str(OUT),'episodes':{k:{a:{q:v.get(q) for q in ['status','controls','progress']} for a,v in c['arms'].items()} for k,c in r['cases'].items()},'global_patch_validated':False},ensure_ascii=False,indent=2));return
 m=prepare()
 if a.stage in ['check','full']:
  verify_pins(m['pins'])
  # This gate is backed by actual test logs/replays, not a copied boolean admission.
  receipt=read(RECORD/'release_cpu_receipt.json');verify(receipt['log'])
  if not receipt['passed'] or receipt['candidate_pins']!={p:h for p,h in m['pins'].items() if p.startswith(str(REPAIR)+'/')}:raise ValueError('Final candidate full CPU regression receipt missing/mismatched')
  replay={k:read(RECORD/(k+'_replay.json')) for k in ['172_v15','172_native','551_original','551_cutin','552_v15']}
  if not replay_gate(replay):raise ValueError('New replay refusal/missing replay blocks expensive GPU')
  subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(LOCAL/'tests'),'-v'],check=True)
  subprocess.run([sys.executable,'-B',str(ROOT/'integrations/autonomous-patch-agent-v8/probe.py'),'--code',str(CANON),'--output',str(OUT/'smpl_preflight.json')],check=True)
  for c in m['cases'].values():
   for work in c['arms'].values():subprocess.run([sys.executable,'-B',str(LOCAL/'pipeline.py'),'_cpu','--work',work],check=True)
  write(OUT/'cpu_admission.json',{'passed':True,'release':asset(OUT/'matrix.json'),'candidate_regression':asset(RECORD/'release_cpu_receipt.json'),'replays_counterfactual_only':True,'known_552_refusal_not_resolved':True,'GPU_executed':False})
 if a.stage=='full':
  import torch
  if not torch.cuda.is_available():print('CUDA unavailable; CPU admission saved; GPU episodes not started',flush=True);return
  with (OUT/'coordinator.lock').open('a') as coordinator,(ROOT/'outputs/gpu_test_service/worker.lock').open('a') as lock:
   fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if not (OUT/'started.json').exists():write(OUT/'started.json',{'epoch':time.time()})
   deadline=read(OUT/'started.json')['epoch']+m['budget']['wall_seconds']
   for case,c in m['cases'].items():
    for arm,w in c['arms'].items():
     work=Path(w)
     if (OUT/'STOP').exists() or (ROOT/'outputs/gpu_test_service/STOP').exists():evaluate(m);return
     if (work/'completion.json').exists():
      done=read(m.get('postprocess_overrides',{}).get(str(work),work/'completion.json'))
      if done['status'] in ['failed','resource_limited'] or done['scoring_error']:raise ValueError('Previous failed episode retained; needs explicit investigation')
      continue
     if (work/c['episode_name']).exists():raise ValueError('Partial episode retained; no action stitching or overwrite: '+w)
     remaining=deadline-time.time()
     if remaining<=0:raise ValueError('Shared validation wall budget exhausted; no implicit reset')
     write(OUT/'progress.json',{'case':case,'arm':arm,'wall_remaining_s':remaining,'stage':'fresh_closed_loop'})
     log=work/'launch.log'
     with log.open('x') as f:
      env=dict(os.environ,BOX_VALIDATION_GPU_LEASE_FD=str(lock.fileno()))
      child=subprocess.Popen([sys.executable,'-u','-B',str(LOCAL/'pipeline.py'),'_execute','--work',w],stdout=f,stderr=subprocess.STDOUT,env=env,pass_fds=(lock.fileno(),),start_new_session=True)
      try:
       code=child.wait(timeout=remaining)
       if code:raise subprocess.CalledProcessError(code,child.args)
      finally:
       if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
       # Also stop any worker descendants that outlived an abnormal driver exit.
       try:os.killpg(child.pid,signal.SIGTERM)
       except ProcessLookupError:pass
       evaluate(m)
 print(json.dumps({'output':str(OUT),'prepared':True,'global_patch_validated':False},ensure_ascii=False))
if __name__=='__main__':main()
