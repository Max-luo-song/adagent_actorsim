"""Private cross-scene gate. Patch-authoring sees only one aggregate verdict."""

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

import copy,contextlib,importlib.util,sys,time
pass  # Path is provided by the portable boundary above
from core import AgentError,BudgetExceeded,ROOT,asset,digest,read,write,verify,verify_pins,utc
import jobs

def policy(config):
 p=config['generalization']
 if p.get('required') is not True or p.get('mode')!='blind_cross_scene' or p.get('feedback_policy')!='binary_only':raise AgentError('Blind cross-scene validation is mandatory')
 if type(p.get('max_development_queries')) is not int or not 1<=p['max_development_queries']<=10:raise AgentError('Bounded private development queries required')
 try:
  verify(p['private_registry']);r=read(p['private_registry']['path'])
  dev=r['development_cases'];sealed=r['sealed_cases'];train=p['training_scene']
  if len({c['scene'] for c in dev})<2 or not sealed:raise ValueError('Missing independent folds')
  if train in {c['scene'] for c in dev+sealed} or {c['scene'] for c in dev}&{c['scene'] for c in sealed}:raise ValueError('Folds overlap')
  if len({c['id'] for c in dev+sealed})!=len(dev+sealed):raise ValueError('Duplicate private cases')
 except (OSError,KeyError,ValueError,AgentError):raise AgentError('Private validation registry missing or invalid; details withheld') from None
 return p

class PrivateAdapter:
 def __init__(self):self.cache={}
 def _adapter(self,case):
  binding=case.get('adapter')
  if not binding:raise AgentError('Private execution adapter unavailable')
  verify(binding['module']);key=binding['module']['sha256']
  if key not in self.cache:
   path=Path(binding['module']['path']).resolve()
   if not path.is_relative_to(ROOT):raise AgentError('Unregistered private adapter')
   spec=importlib.util.spec_from_file_location('_private_gate_'+key[:12],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   self.cache[key]=getattr(module,binding['factory'])(copy.deepcopy(case))
  return self.cache[key]
 def generalization_stage(self,case,candidate,folder,config):return self._adapter(case).generalization_stage(case,candidate,folder,config)
 def generalization_result(self,case,folder,receipt):return self._adapter(case).generalization_result(case,folder,receipt)

def public_feedback(state):
 status=(state.get('blind_repair') or {}).get('public_feedback')
 return dict(status='not_passed' if status=='not_passed' else 'not_evaluated',instruction='Inspect mechanisms using training-scene evidence and new differential/metamorphic tests. No validation scene, score, cause, trace or recommendation is available.',validation_details_available=False)

def return_to_training(state,root):
 root=Path(root);task=state['active_task_id'];ledger=state['task_convergence'][task]
 folder=root/'training_review_epochs'/('epoch_'+str(len(list((root/'training_review_epochs').glob('epoch_*')))+1).zfill(3))
 write(folder/'convergence_before.json',ledger)
 ledger['window']=[];ledger['small_gain_streak']=0;ledger['converged']=False
 state.update(status='ready',task_switch_pending=False,generalization_pending=False,final_patch_status='provisional',registered_generalization_validated=False)
 state.pop('accepted_patch',None);state.pop('generalization',None)
 state['blind_repair']=dict(active=True,audit_required=True,audit_reserved=False,public_feedback='not_passed',require_new_causal_test=True,training_only=True)
 write(folder/'review.json',public_feedback(state))

def training_verdict(verdict,state,metrics,policy):
 # Equal-score robust repairs are admissible only after a private rejection;
 # every safety, progress, original and protected-frontier condition remains.
 result=copy.deepcopy(verdict)
 if not (state.get('blind_repair') or {}).get('active'):return result
 target=state['task_convergence'][state['active_task_id']]['policy']['target_case'];n=metrics[target]['score']['hdscore'];b=state['champion']['metrics'][target]['score']['hdscore'];tol=policy['numeric_tolerance']
 if n>=b-tol:
  result['reasons']=[x for x in result['reasons'] if x!=target+':no_incremental_target_gain'];result['accepted']=not result['reasons']
  result['robustness_revision']=bool(result['accepted']);result['local_score_improvement_claim']=n>b+tol
 return result

def _ready(cases):return all(c.get('ready') is True and c.get('baseline') and c.get('adapter') for c in cases)

def _publish(state,config,root,registry,private,meta):
 import generalization as g
 final=read(private/'current.json');shadow=read(Path(final['folder'])/'state.json');request=read(shadow['generalization']['request']['path'])
 checked=[g.verify_case(r['proof'],state['champion'],c,request['requested_epoch'],request['score_tolerance']) for c,r in zip(request['cases'],shadow['generalization']['results'])]
 if request['cases']!=registry['sealed_cases'] or state.get('simulation_only') or len(checked)!=len(registry['sealed_cases']) or not checked or any(not r['passed'] for r in checked):raise AgentError('Private final validation incomplete')
 # Details remain exclusively in private storage; the public package is an attestation.
 proof=private/'final_attestation.json';write(proof,dict(candidate_pins=state['champion']['code_pins'],results=checked,registry=config['generalization'].get('pool_catalog',config['generalization'].get('private_registry')),confirmed=True,time=utc()))
 package=dict(schema='confirmed_blind_cross_scene_patch_v1',candidate=state['champion'],confirmed=True,registered_generalization_validated=True,cross_scene_generalization_validated=True,validation_scope=config['generalization']['mode'],private_attestation=asset(proof),formal_certification=False,scope='Only registered non-training scenes; identities and individual results withheld from patch author',time=utc())
 import fcntl
 pointer=Path(config['generalization']['global_parent']['path']);public=Path(root)/'accepted_patch.json'
 with (pointer.parent/'global_patch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);current=read(pointer)
  if public.exists() and current.get('accepted_patch')==asset(public) and read(public).get('candidate',{}).get('code_pins')==state['champion']['code_pins']:
   accepted=asset(public)
  else:
   verify(config['generalization']['global_parent']);previous=private/'previous_global.json';write(previous,current)
   write(public,package);accepted=asset(public)
   write(pointer,dict(schema='confirmed_blind_cross_scene_default_v1',version=state['champion']['version'],policy=state['champion'],accepted_patch=accepted,confirmed_default=True,user_confirmed_default=False,default_for_new_tasks=True,registered_generalization_validated=True,cross_scene_generalization_validated=True,validation_scope=config['generalization']['mode'],previous_global=asset(previous)))
 state.update(accepted_patch=accepted,final_patch_status='confirmed',registered_generalization_validated=True,generalization_pending=False)
 state['blind_repair']={'active':False};state['status']='generalization_passed';meta['confirmed']=True;meta['confirmed_candidate']=digest(state['champion']['code_pins']);meta['confirmed_task']=state['active_task_id']

def step(state,config,adapter,root):
 import generalization as g
 p=policy(config);root=Path(root);private=root/'private_validation';private.mkdir(parents=True,exist_ok=True)
 try:private.chmod(0o700)
 except OSError:pass
 registry=read(p['private_registry']['path']);meta_file=private/'meta.json'
 meta=read(meta_file) if meta_file.exists() else dict(registry=p['private_registry'],development_queries=0,sealed_consumed=False,tested_controllers=[],current=None,confirmed=False)
 if meta['registry']!=p['private_registry']:state['status']='waiting_private_registry_revision';return False
 if meta.get('confirmed'):
  public=root/'accepted_patch.json'
  if meta.get('confirmed_candidate')==digest(state['champion']['code_pins']) and meta.get('confirmed_task')==state['active_task_id'] and public.exists() and state.get('accepted_patch')==asset(public):return True
  state['status']='waiting_fresh_private_validation_pool';return False
 if not state.get('task_convergence',{}).get(state['active_task_id'],{}).get('converged'):raise AgentError('Convergence required before private validation')
 fingerprint=digest(state['champion']['code_pins'])
 if state['champion']['code_pins']==registry['baseline_controller_pins']:state['status']='waiting_new_training_patch';return False
 if not meta.get('current'):
  consumed=Path(p['private_registry']['path']).parent/'sealed_consumption.json'
  if consumed.exists() and read(consumed).get('campaign')!=str(root.resolve()):state['status']='waiting_fresh_private_validation_pool';return False
  if meta['sealed_consumed']:state['status']='waiting_fresh_private_validation_pool';return False
  if meta['development_queries']>=p['max_development_queries']:state['status']='private_validation_query_budget_exhausted';return False
  if fingerprint in meta['tested_controllers']:state['status']='waiting_new_training_patch';return False
  if not _ready(registry['development_cases']) or not _ready(registry['sealed_cases']):state['status']='waiting_private_validation_assets';return False
  try:
   for case in registry['development_cases']+registry['sealed_cases']:
    verify(case['baseline'])
    if read(case['baseline']['path'])['code_pins']!=registry['baseline_controller_pins']:raise ValueError('Private baseline controller mismatch')
  except (OSError,KeyError,ValueError,AgentError):state['status']='private_validation_admission_error';return False
  meta['development_queries']+=1;meta['tested_controllers'].append(fingerprint)
  meta['current']=dict(stage='development',folder=str(private/('attempt_%03d'%meta['development_queries'])/'development'),fingerprint=fingerprint)
 current=meta['current']
 if current['fingerprint']!=fingerprint:raise AgentError('Candidate changed during private validation')
 cases=registry['development_cases'] if current['stage']=='development' else registry['sealed_cases']
 folder=Path(current['folder']);shadow_file=folder/'state.json'
 internal=copy.deepcopy(config);internal['generalization']=dict(required=True,mode='cross_scene',cases=cases,score_tolerance=1e-9,_host_single_sealed_scene=current['stage']=='sealed')
 internal['_host_collect_all_validation_cases']=True;internal['_host_stop_path']=str(root/'STOP')
 shadow=read(shadow_file) if shadow_file.exists() else dict(champion=copy.deepcopy(state['champion']),active_task_id='private_gate',simulation_only=state.get('simulation_only',False),started_epoch=state['started_epoch'],rounds_started=state['rounds_started'],usage=copy.deepcopy(state['usage']))
 shadow['usage']=copy.deepcopy(state['usage']);write(private/'current.json',current);write(meta_file,meta)
 try:
  with (private/'host_evaluator.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   result=g.step(shadow,internal,PrivateAdapter(),folder)
 except BudgetExceeded:
  raise BudgetExceeded('Shared validation budget exhausted') from None
 except (Exception,SystemExit) as exc:
  write(folder/'private_error.json',dict(error_type=type(exc).__name__,error=str(exc)));state['status']='private_validation_infrastructure_error';return False
 finally:
  state['usage']=copy.deepcopy(shadow['usage']);write(shadow_file,shadow);write(root/'state.json',state)
 if result is None:state['status']='running_private_validation';return None
 if result is False:
  if (shadow.get('generalization') or {}).get('status')!='failed':state['status']='private_validation_infrastructure_error';return False
  if current['stage']=='sealed':meta['sealed_consumed']=True
  meta['current']=None;write(meta_file,meta);return_to_training(state,root);return None
 if current['stage']=='development':
  import fcntl
  registry_folder=Path(p['private_registry']['path']).parent;consumption=registry_folder/'sealed_consumption.json'
  with (registry_folder/'sealed.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   identity=dict(campaign=str(root.resolve()),candidate=fingerprint,registry=p['private_registry'])
   if consumption.exists() and read(consumption)!=identity:state['status']='waiting_fresh_private_validation_pool';return False
   if not consumption.exists():write(consumption,identity)
  meta['current']=dict(stage='sealed',folder=str(private/('attempt_%03d'%meta['development_queries'])/'sealed'),fingerprint=fingerprint);meta['sealed_consumed']=True;write(meta_file,meta);state['status']='running_private_validation';return None
 try:
  with (private/'host_evaluator.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):_publish(state,config,root,registry,private,meta)
 except (Exception,SystemExit) as exc:
  write(private/'private_publication_error.json',dict(error_type=type(exc).__name__,error=str(exc)));state['status']='private_validation_infrastructure_error';return False
 write(meta_file,meta);return True

def cancel(root):
 private=Path(root)/'private_validation';current=private/'current.json'
 if current.exists():
  state=Path(read(current)['folder'])/'state.json'
  if state.exists():
   job=(read(state).get('generalization') or {}).get('current_job')
   if job:jobs.cancel(Path(job))
