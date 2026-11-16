"""Consume a failed validation pool BEFORE releasing it as training evidence."""

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

import copy,contextlib,fcntl,time
import jobs
pass  # Path is provided by the portable boundary above
from core import AgentError,BudgetExceeded,asset,digest,read,write,verify,verify_pins,utc,remaining
from blind_validation import PrivateAdapter

MODE='counterexample_cross_scene'

def policy(config):
 p=config['generalization']
 if p.get('mode')!=MODE or p.get('required') is not True or p.get('feedback_policy')!='retire_then_release':raise AgentError('Counterexample validation lifecycle required')
 try:verify(p['pool_catalog'])
 except (OSError,AgentError):raise AgentError('Private validation catalog unavailable') from None
 return p

def load_ledger(config):
 folder=Path(config['generalization']['pool_catalog']['path']).parent;folder.mkdir(parents=True,exist_ok=True)
 file=folder/'dataset_lifecycle.json'
 return file,read(file) if file.exists() else dict(schema='validation_to_training_lifecycle_v1',pools={},consumed_scenes=[],training_scenes=[config['generalization']['training_scene']])

def choose_pool(state,config):
 p=policy(config);file,ledger=load_ledger(config);catalog=read(p['pool_catalog']['path'])
 train=set(ledger['training_scenes'])|set(state.get('training_scene_history',[]))|{state.get('current_task_descriptor',{}).get('scene',p['training_scene'])}
 excluded=set(ledger['consumed_scenes'])|train|{scene for record in ledger['pools'].values() for scene in record['scene_ids']}
 for binding in catalog['pools']:
  verify(binding);pool=read(binding['path']);identity=digest(binding)
  if identity in ledger['pools']:continue
  scenes={c['scene'] for c in pool['cases']}
  if len(scenes)<2 or scenes&excluded:continue
  if len({c['id'] for c in pool['cases']})!=len(pool['cases']):raise AgentError('Duplicate validation case')
  if config.get('host_governance',{}).get('enabled'):
   from pool_stratification import validate
   try:validate(pool,config['host_governance']['stratification'])
   except AgentError:continue
  return binding,pool
 return None,None

def _cause(side):
 # Full details are read only after the entire originating pool is retired.
 report=read(side['report']['path']);summary=read(side['summary']['path'])
 source=report.get('source_report')
 if source:verify(source);report=read(source['path'])
 return dict(status=report.get('status'),termination_reason=report.get('termination_reason'),error=report.get('error'),error_stage=summary.get('error_stage'),controls=len(report.get('trace',[])),road_safety_termination=report.get('road_safety_termination'),last_observation=(report.get('observations') or [None])[-1],report=side['report'],summary=side['summary'],contract=side['contract'],source_report=source)

def retire_and_release(state,config,root,pool_binding,pool,request,results):
 root=Path(root);ledgerfile,ledger=load_ledger(config);identity=digest(pool_binding);folder=root/'released_counterexamples'/identity[:16];folder.mkdir(parents=True,exist_ok=True)
 import generalization as g
 if request['cases']!=pool['cases'] or len(results)!=len(pool['cases']):raise AgentError('Complete matched validation results required for retirement')
 checked=[g.verify_case(r['proof'],state['champion'],c,request['requested_epoch'],request['score_tolerance'],config) for c,r in zip(pool['cases'],results)]
 if checked!=results or not any(not r['passed'] for r in checked):raise AgentError('Retirement requires verified control rejection, not infrastructure failure')
 # Commit dataset retirement first; crash recovery may replay publication only.
 with ledgerfile.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(ledgerfile) if ledgerfile.exists() else ledger
  old=ledger['pools'].get(identity)
  if old and (old.get('status') not in ('evaluating','retired_failed') or old.get('candidate_pins')!=state['champion']['code_pins']):raise AgentError('Consumed validation pool cannot be rebound')
  ledger['pools'][identity]=dict(old or {},status='retired_failed',candidate_pins=state['champion']['code_pins'],scene_ids=sorted({c['scene'] for c in pool['cases']}),pool=pool_binding)
  ledger['pools'][identity].setdefault('retired_at',utc())
  ledger['consumed_scenes']=sorted(set(ledger['consumed_scenes'])|{c['scene'] for c in pool['cases']});write(ledgerfile,ledger)
 if (folder/'release.json').exists():
  release=asset(folder/'release.json');package=verify_release(release,config)
  _queue_release(state,release,package);return release
 evidence=[];released_paths=[]
 if request['cases']!=pool['cases'] or len(results)!=len(pool['cases']) or any(r.get('case_id')!=c['id'] for c,r in zip(pool['cases'],results)):raise AgentError('Complete matched validation results required for release')
 for case,result in zip(request['cases'],results):
  if result['passed']:continue
  verify(result['proof']);proof=read(result['proof']['path']);side=proof['candidate']
  cause=_cause(side);entry=dict(case=case,proof=result['proof'],failure=result,details=cause,classification='retired_validation_training_counterexample')
  evidence.append(entry)
  for key in ('report','summary','contract'):
   released_paths.append(str(Path(side[key]['path']).resolve().parent))
  if cause.get('source_report'):released_paths.append(str(Path(cause['source_report']['path']).resolve().parent))
 if not evidence:raise AgentError('No actual rejected case to release')
 package=dict(schema='retired_validation_release_v1',pool=pool_binding,pool_identity=identity,pool_status='retired_failed',retirement_ledger=asset(ledgerfile),candidate=copy.deepcopy(state['champion']),failed_cases=evidence,released_paths=sorted(set(released_paths)),previous_training_task=state.get('current_task_descriptor'),validation_reuse_forbidden=True,role='training_only_not_generalization_evidence',time=utc())
 write(folder/'release.json',package);release=asset(folder/'release.json')
 with ledgerfile.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(ledgerfile);ledger['pools'][identity]['release']=release;write(ledgerfile,ledger)
 _queue_release(state,release,package);return release

def _queue_release(state,release,package):
 state.setdefault('released_counterexample_manifests',[])
 if release not in state['released_counterexample_manifests']:state['released_counterexample_manifests'].append(release)
 done=set(state.get('counterexample_promotions',{}))
 queued={(digest(x['release']),x['index']) for x in state.get('pending_counterexamples',[])}
 for i in range(len(package['failed_cases'])):
  key=digest(dict(release=release,index=i))
  if key not in done and (digest(release),i) not in queued:state.setdefault('pending_counterexamples',[]).append(dict(release=release,index=i))
 state.update(status='counterexample_training_pending',task_switch_pending=True,final_patch_status='provisional',registered_generalization_validated=False,generalization_pending=False)
 state.pop('accepted_patch',None)

def released_paths(config,bindings):
 paths=[]
 for binding in bindings:
  release=verify_release(binding,config)
  # Derive exceptions from authenticated failed proofs, never trust arbitrary paths.
  for failure in release['failed_cases']:
   verify(failure['proof']);proof=read(failure['proof']['path'])
   for key in ('report','summary','contract'):
    verify(proof['candidate'][key]);paths.append(str(Path(proof['candidate'][key]['path']).resolve()))
   source=failure['details'].get('source_report')
   if source:verify(source);paths.append(str(Path(source['path']).resolve()));paths.append(str(Path(source['path']).resolve().parent/'steps'))
 return sorted(set(paths))

def verify_release(binding,config):
 verify(binding);release=read(binding['path']);_,ledger=load_ledger(config);record=ledger['pools'].get(release['pool_identity'])
 verify(release['pool'])
 if record and record.get('pool')!=release.get('pool'):raise AgentError('Release pool binding differs from retired record')
 if release.get('pool_status')!='retired_failed' or not record or record['status']!='retired_failed' or not release.get('validation_reuse_forbidden'):raise AgentError('Validation details cannot be revealed before pool retirement')
 return release

def promote_pending(state,config,root):
 queue=state.get('pending_counterexamples') or []
 if not queue:return False
 pending=queue[0];release=verify_release(pending['release'],config);failure=copy.deepcopy(release['failed_cases'][pending['index']]);failure['release']=pending['release'];case=failure['case']
 adapter=PrivateAdapter()._adapter(case);compiler=getattr(adapter,'compile_counterexample_training',None)
 # The trusted scene adapter compiles matched real baselines and regression assets;
 # it may return None while registered asynchronous work is pending.
 folder=Path(root)/'counterexample_training'/(digest(case)[:12]+'_'+digest(state['champion']['code_pins'])[:12])
 promotion=compiler(copy.deepcopy(failure),copy.deepcopy(state['champion']),copy.deepcopy(state.get('task_archive',[])),folder,copy.deepcopy(config)) if compiler else None
 if promotion is None:
  stage_factory=getattr(adapter,'counterexample_training_stage',None);reader=getattr(adapter,'counterexample_training_result',None)
  if stage_factory is None or reader is None:state['status']='waiting_counterexample_training_admission';return False
  stage=stage_factory(copy.deepcopy(failure),copy.deepcopy(state['champion']),copy.deepcopy(state.get('task_archive',[])),folder,copy.deepcopy(config));verify(stage['admission']);admission=read(stage['admission']['path'])
  if not admission.get('passed') or admission.get('case_id')!=case['id'] or admission.get('candidate_pins')!=state['champion']['code_pins'] or admission.get('release')!=pending['release']:raise AgentError('Counterexample preparation not bound to retired case/controller')
  job=folder/'job';reservations=state.setdefault('counterexample_preparation_reservations',[])
  if str(job) not in reservations:
   if state['usage'].get('gpu_attempts',0)>=config['budgets']['max_gpu_attempts']:raise BudgetExceeded('Counterexample preparation shares GPU budget')
   state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;state['usage']['gpu_dispatches']=state['usage'].get('gpu_dispatches',0)+1;reservations.append(str(job));write(Path(root)/'state.json',state)
  inputs={**stage['inputs'],**state['champion']['code_pins'],stage['admission']['path']:stage['admission']['sha256'],pending['release']['path']:pending['release']['sha256']}
  receipt=jobs.launch(job,stage['command'],stage.get('cwd',root),min(stage['timeout'],remaining(state,config)),inputs,stage['outputs'],Path(root)/'STOP');state['counterexample_preparation_job']=str(job)
  if receipt['status']=='running':state['status']='preparing_counterexample_training';return None
  if receipt['status']!='succeeded':state['status']='counterexample_preparation_infrastructure_error';return False
  promotion=reader(copy.deepcopy(failure),folder,receipt);state.pop('counterexample_preparation_job',None)
  if not promotion:raise AgentError('Counterexample preparation returned no admitted baseline')
 from task_rotation import activate_counterexample
 activate_counterexample(state,config,promotion,pending['release'],case,root)
 queue.pop(0);state['pending_counterexamples']=queue;state['adapter_switch_pending']=True
 return None

def step(state,config,adapter,root):
 import generalization as g
 root=Path(root);p=policy(config)
 if state.get('pending_counterexamples'):
  try:return promote_pending(state,config,root)
  except (AgentError,OSError,KeyError,TypeError,ValueError):state['status']='waiting_counterexample_training_admission';return False
 if not state['task_convergence'][state['active_task_id']]['converged']:raise AgentError('Training convergence required before validation')
 private=root/'private_validation';private.mkdir(parents=True,exist_ok=True);private.chmod(0o700);current=private/'active_pool.json'
 if current.exists():binding=read(current);verify(binding);pool=read(binding['path'])
 else:
  binding,pool=choose_pool(state,config)
  if binding is None:state['status']='waiting_fresh_validation_pool';return False
  write(current,binding)
 file,ledger=load_ledger(config);identity=digest(binding)
 if identity in ledger['pools']:
  if ledger['pools'][identity]['status']=='retired_failed':
   shadow=read(private/identity[:16]/'state.json');request=read(shadow['generalization']['request']['path'])
   retire_and_release(state,config,root,binding,pool,request,shadow['generalization']['results']);current.unlink();return None
  if ledger['pools'][identity]['status']!='evaluating':
   if state.get('final_patch_status')=='confirmed':return True
   state['status']='waiting_fresh_validation_pool';return False
  if ledger['pools'][identity].get('owner')!=str(root.resolve()) or ledger['pools'][identity]['candidate_pins']!=state['champion']['code_pins']:raise AgentError('Validation pool belongs to another candidate/campaign')
 if any(not c.get('ready') or not c.get('baseline') or not c.get('adapter') for c in pool['cases']):state['status']='waiting_cross_scene_assets';return False
 for case in pool['cases']:
  verify(case['baseline'])
  if read(case['baseline']['path']).get('code_pins')!=pool['baseline_controller_pins']:raise AgentError('Validation baseline is not the frozen global parent')
 if state['champion']['code_pins']==pool['baseline_controller_pins']:state['status']='waiting_new_training_patch';return False
 if config.get('host_governance',{}).get('enabled'):
  from search_governance import admit_validation_pool
  admission=admit_validation_pool(state,config,digest(binding),len(pool['cases']))
  if not admission['passed']:state['status']=admission['reason'];return False
 with file.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(file) if file.exists() else ledger
  record=ledger['pools'].get(identity)
  scenes={c['scene'] for c in pool['cases']}
  if record is None:
   if scenes&set(ledger['consumed_scenes']) or any(scenes&set(x['scene_ids']) for x in ledger['pools'].values()):state['status']='waiting_fresh_validation_pool';return False
   ledger['pools'][identity]=dict(status='evaluating',owner=str(root.resolve()),candidate_pins=state['champion']['code_pins'],scene_ids=sorted(scenes),pool=binding);write(file,ledger)
  elif record.get('owner')!=str(root.resolve()) or record['candidate_pins']!=state['champion']['code_pins']:raise AgentError('Validation reservation changed')
 folder=private/identity[:16];shadow_file=folder/'state.json'
 internal=copy.deepcopy(config);internal['generalization']=dict(required=True,mode='cross_scene',cases=pool['cases'],score_tolerance=1e-9);internal['_host_collect_all_validation_cases']=True;internal['_host_stop_path']=str(root/'STOP')
 shadow=read(shadow_file) if shadow_file.exists() else dict(champion=copy.deepcopy(state['champion']),active_task_id='private_gate',simulation_only=state.get('simulation_only',False),started_epoch=state['started_epoch'],rounds_started=state['rounds_started'],usage=copy.deepcopy(state['usage']))
 if shadow['champion']['code_pins']!=state['champion']['code_pins']:raise AgentError('Pool is frozen to a different candidate')
 shadow['usage']=copy.deepcopy(state['usage']);write(private/'current.json',dict(folder=str(folder)))
 try:
  with (private/'host_evaluator.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=g.step(shadow,internal,PrivateAdapter(),folder)
 except BudgetExceeded:raise BudgetExceeded('Shared validation budget exhausted') from None
 except (Exception,SystemExit) as exc:write(folder/'private_error.json',dict(error_type=type(exc).__name__,error=str(exc)));state['status']='validation_infrastructure_error';return False
 finally:state['usage']=copy.deepcopy(shadow['usage']);write(shadow_file,shadow);write(root/'state.json',state)
 if result is None:state['status']='running_cross_scene_validation';return None
 if result is False:
  if (shadow.get('generalization') or {}).get('status')!='failed':state['status']='validation_infrastructure_error';return False
  request=read(shadow['generalization']['request']['path']);retire_and_release(state,config,root,binding,pool,request,shadow['generalization']['results'])
  current.unlink();return None
 # The existing private publisher revalidates every actual proof before writing a
 # public attestation; this pool is then consumed for any future candidate.
 from blind_validation import _publish
 request=read(shadow['generalization']['request']['path'])
 try:
  with (private/'host_evaluator.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):_publish(state,config,root,{'sealed_cases':pool['cases']},private,{})
 except (Exception,SystemExit) as exc:
  write(folder/'private_publish_error.json',dict(error_type=type(exc).__name__,error=str(exc)));state['status']='validation_publish_infrastructure_error';return False
 ledger['pools'][identity]=dict(status='accepted_consumed',candidate_pins=state['champion']['code_pins'],scene_ids=sorted({c['scene'] for c in pool['cases']}),pool=binding,time=utc())
 ledger['consumed_scenes']=sorted(set(ledger['consumed_scenes'])|{c['scene'] for c in pool['cases']});write(file,ledger);return True
