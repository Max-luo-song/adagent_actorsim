"""Host-owned task queue; unsupported/unadmitted tasks wait, never borrow scene assets."""

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

import copy,importlib.util,sys
pass  # Path is provided by the portable boundary above
from core import ROOT,AgentError,asset,read,verify,verify_pins,write,sha,utc,remaining
import convergence

def load_adapter(task,config,checkpoint):
 binding=task.get('adapter')
 if not isinstance(binding,dict) or set(binding)!={'module','factory'}:raise AgentError('Registered task adapter binding required')
 verify(binding['module']);path=Path(binding['module']['path']).resolve()
 if not path.is_relative_to(ROOT) and not checkpoint.get('simulation_only'):raise AgentError('Task adapter must be a registered project module')
 spec=importlib.util.spec_from_file_location('_registered_task_'+binding['module']['sha256'][:16],path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 factory=getattr(module,binding['factory']);adapter=factory(copy.deepcopy(task),copy.deepcopy(config),copy.deepcopy(checkpoint))
 for name in ('stages','build','attempt','results','validate_stage','classify_failure','activate_task','verify_task_admission','acceptance','collect_evidence','retention_check'):
  if not callable(getattr(adapter,name,None)):raise AgentError('Task adapter missing contract method: '+name)
 return adapter

def current_id(config):return config['task_queue']['current_id']
def initialize(state,config):
 if not config.get('convergence',{}).get('enabled'):return
 task=current_id(config)
 spec=next(x for x in config['task_queue']['tasks'] if x['id']==task)
 if not state['simulation_only']:
  for m in state['champion']['metrics'].values():
   if read(m['contract']['path']).get('scene')!=spec['scene']:raise AgentError('Initial task scene differs from baseline contracts')
 state['active_task_id']=task;state['current_task_descriptor']=copy.deepcopy(spec);state['task_archive']=[];state['task_convergence']={task:convergence.initial(task,state['champion']['metrics'],config['convergence'],state['simulation_only'])}

def record(state,rid,metrics,verdict,best_before,root):
 task=state['active_task_id'];ledger=state['task_convergence'][task];row=convergence.observe(ledger,rid,metrics,verdict,best_before,state['simulation_only'])
 write(Path(root)/'rounds'/rid/'convergence.json',dict(observation=row,small_gain_streak=ledger['small_gain_streak'],converged=ledger['converged'],policy=ledger['policy'],task_id=task))
 write(Path(root)/'convergence.json',dict(active_task_id=task,tasks=state['task_convergence']))
 if ledger['converged']:
  state.update(status='task_converged',task_switch_pending=True)
  if True:
   archive=dict(confirmed=False,status='provisional_converged_candidate',task_id=task,convergence=copy.deepcopy(ledger),champion=copy.deepcopy(state['champion']),registry=copy.deepcopy(state.get('registry')),reference_metrics=copy.deepcopy(state['reference_metrics']),task_spec=copy.deepcopy(state.get('active_task_spec') or state.get('current_task_descriptor')),time=utc());
   existing=next((i for i,x in enumerate(state['task_archive']) if x['task_id']==task),None)
   if existing is None:state['task_archive'].append(archive)
   else:
    previous=state['task_archive'][existing];write(Path(root)/'task_archive'/task/'previous_convergences'/('%03d.json'%len(ledger['observations'])),previous);state['task_archive'][existing]=archive
   write(Path(root)/'task_archive'/task/'summary.json',archive)

def remaining_budget(state,config):
 b=config['budgets'];u=state['usage']
 return dict(wall_seconds=remaining(state,config),rounds=b['max_rounds']-state['rounds_started'],api_requests=b['max_api_requests']-u['api_requests'],tokens=b['max_api_tokens']-u['charged_tokens'],gpu_attempts=b['max_gpu_attempts']-u.get('gpu_attempts',0))

def advance(state,config,adapter,root):
 root=Path(root);task=state['active_task_id'];ledger=state['task_convergence'][task]
 if not ledger['converged']:raise AgentError('Cannot rotate an unconverged task')
 if not (state.get('simulation_only') and config.get('generalization_fixture_bypass') is True):
  if state.get('final_patch_status')!='confirmed' or not state.get('accepted_patch'):raise AgentError('Generalization confirmation required before task rotation')
  verify(state['accepted_patch'])
 verify_pins(state['champion']['code_pins'])
 if task not in [x['task_id'] for x in state['task_archive']]:
  archive=dict(confirmed=False,status='provisional_converged_candidate',task_id=task,convergence=copy.deepcopy(ledger),champion=copy.deepcopy(state['champion']),registry=copy.deepcopy(state.get('registry')),reference_metrics=copy.deepcopy(state['reference_metrics']),task_spec=copy.deepcopy(state.get('active_task_spec') or state.get('current_task_descriptor')),time=utc())
  state['task_archive'].append(archive);write(root/'task_archive'/task/'summary.json',archive)
 done={x['task_id'] for x in state['task_archive']};queue=config['task_queue']['tasks'];candidates=[x for x in queue if x['id'] not in done and x.get('ready') and (not config.get('scene_cycle',{}).get('different_scene_after_confirmation') or x['scene']!=state['current_task_descriptor']['scene'])]
 next_task=candidates[0] if candidates else None;budget=remaining_budget(state,config)
 transition=dict(from_task=task,next_task=next_task,carry_champion=state['champion'],budget_remaining=budget,completed_tasks=sorted(done),unavailable_tasks=[x for x in queue if x['id'] not in done and not x.get('ready')],full_fresh_baseline_required=True,previous_task_nonregression_required=True,budget_reset=False)
 write(root/'next_task_request.json',transition)
 if config.get('scene_cycle',{}).get('different_scene_after_confirmation'):write(root/'next_scene_request.json',dict(transition,accepted_patch=state.get('accepted_patch'),require_different_scene=True,discovery_required=True))
 next_adapter=load_adapter(next_task,config,state) if next_task and next_task.get('adapter') else adapter
 if not next_task or not hasattr(next_adapter,'activate_task'):
  state.update(status='waiting_next_task',task_switch_pending=True,task_wait_reason='No admitted next scene/edit with a compatible adapter and fresh carried-policy baseline')
  return False,adapter
 if any(budget[k]<=0 for k in ('rounds','api_requests','tokens','gpu_attempts')):state.update(status='budget_exhausted',reason='Shared task-rotation budget exhausted');return False,adapter
 # activate_task loads an already admitted, fresh baseline; it may not run GPU/API work.
 activation=next_adapter.activate_task(copy.deepcopy(next_task),copy.deepcopy(state['champion']),copy.deepcopy(state['task_archive']),copy.deepcopy(budget))
 required={'task_id','admission','champion','reference_metrics','registry','target_case','validated'}
 if not isinstance(activation,dict) or set(activation)!=required or activation['task_id']!=next_task['id'] or not activation['validated']:raise AgentError('Task admission incomplete')
 verify(activation['admission']);admission=read(activation['admission']['path']);champ=activation['champion'];verify_pins(champ['code_pins'])
 if not state['simulation_only']:
  policy=Path(state['champion']['code'])/'joint_safety.py'
  if admission.get('carried_policy_sha256')!=sha(policy) or sha(Path(champ['code'])/'joint_safety.py')!=sha(policy):raise AgentError('Task switch must carry the accepted controller unchanged')
  if admission.get('scene')!=next_task['scene'] or admission.get('edit')!=next_task['edit'] or not admission.get('fresh_baseline'):raise AgentError('New scene/edit baseline is not matched')
  for m in champ['metrics'].values():
   for key in ('report','summary','contract'):verify(m[key])
   contract=read(m['contract']['path'])
   if contract.get('scene')!=next_task['scene']:raise AgentError('Task baseline contract belongs to a different scene')
 if not hasattr(next_adapter,'verify_task_admission') or not next_adapter.verify_task_admission(next_task,activation):raise AgentError('Registered adapter did not verify task baseline')
 policy=copy.deepcopy(config['convergence']);policy['target_case']=activation['target_case'];new_ledger=convergence.initial(next_task['id'],champ['metrics'],policy,state['simulation_only'])
 state.update(active_task_id=next_task['id'],active_task_spec=next_task,current_task_descriptor=copy.deepcopy(next_task),champion=champ,reference_metrics=activation['reference_metrics'],task_switch_pending=False,status='ready',diagnostic_cache={})
 if activation['registry'] is None:state.pop('registry',None)
 else:state['registry']=activation['registry']
 state['task_case_ids']=list(champ['metrics'])
 state['task_convergence'][next_task['id']]=new_ledger
 write(root/'task_transitions'/(task+'_to_'+next_task['id']+'.json'),dict(**transition,activation=activation,completed=True))
 write(root/'working_champion.json',champ)
 return True,next_adapter

def verify_retention(adapter,bundle,attempt,archives,policy,simulation_only,host_config=None):
 check=getattr(adapter,'retention_check',None)
 if check is None:return False
 proof=check(bundle,attempt,archives)
 if simulation_only:return proof is True or isinstance(proof,dict) and proof.get('passed') is True
 if not isinstance(proof,dict) or proof.get('passed') is not True or proof.get('candidate_pins')!=bundle['candidate_pins']:return False
 cases=proof.get('tasks',{})
 for old in archives:
  task=old['task_id'];result=cases.get(task)
  if not isinstance(result,dict) or not result.get('evaluation') or not result.get('metrics'):return False
  verify(result['evaluation']);receipt=read(result['evaluation']['path'])
  if receipt.get('candidate_pins')!=bundle['candidate_pins'] or receipt.get('task_id')!=task:return False
  candidate=result['metrics'];baseline=old['champion']['metrics']
  if host_config and host_config.get('host_governance',{}).get('enabled'):
   from trajectory_quality import for_metrics
   if not for_metrics(candidate,host_config,False)['passed']:return False
  if set(candidate)!=set(baseline) or convergence.fingerprint(task,candidate,False)!=old['convergence']['experiment_fingerprint']:return False
  for name,n in candidate.items():
   b=baseline[name]
   for key in ('report','summary','contract'):verify(n[key])
   if n['status']!='completed' or n['controls']!=b['controls'] or n['observations']!=b['observations'] or any(n[k] for k in ('collision','map_departure','unresolved_stop')):return False
   if any(not convergence.finite(n.get('score',{}).get(k)) or not convergence.finite(b.get('score',{}).get(k)) or n['score'][k]<b['score'][k]-policy['numeric_tolerance'] for k in ('hdscore','pdms','rc')):return False
   if any(not convergence.finite(n[k]) for k in ('progress_m','stopped_s','final_abs_lateral_m','peak_abs_lateral_m')):return False
   if n['progress_m']<b['progress_m']*policy['min_progress_ratio'] or n['stopped_s']>b['stopped_s']+policy['max_extra_stopped_seconds']:return False
   for key in (() if policy.get('lateral_diagnostics_only',False) else ('final_abs_lateral_m','peak_abs_lateral_m')):
    ref=old['champion']['reference_metrics'][name][key]
    if not convergence.finite(n[key]) or n[key]>min(b[key],ref)+policy['lateral_tolerance_m']:return False
 return True


def verify_counterexample_task(state,config):
 from validation_lifecycle import verify_release
 binding=state.get('counterexample_task_admission')
 if not binding:raise AgentError('Dynamic task has no counterexample admission')
 verify(binding);proof=read(binding['path']);verify_release(proof['release'],config)
 if proof['task']!=state.get('active_task_spec'):raise AgentError('Dynamic task differs from admitted task')
 verify(proof['task']['adapter']['module'])
 return proof

def activate_counterexample(state,config,promotion,release,case,root):
 from validation_lifecycle import verify_release
 released=verify_release(release,config)
 if case not in [f['case'] for f in released['failed_cases']]:raise AgentError('Task not in released failures')
 task=copy.deepcopy(promotion['task']);activation=promotion['activation']
 if task['scene']!=case['scene'] or not task.get('ready'):raise AgentError('Counterexample scene/task not admitted')
 if task['id'] in state['task_convergence']:raise AgentError('Counterexample task ID already used')
 if activation.get('task_id')!=task['id'] or not activation.get('validated'):raise AgentError('Counterexample activation incomplete')
 verify(activation['admission']);admission=read(activation['admission']['path']);champ=activation['champion'];verify_pins(champ['code_pins'])
 if champ['code_pins']!=state['champion']['code_pins']:raise AgentError('Counterexample must carry rejected candidate unchanged')
 if admission.get('release')!=release or admission.get('scene')!=case['scene'] or admission.get('edit')!=task['edit'] or not admission.get('fresh_baseline'):raise AgentError('Counterexample baseline/release mismatch')
 if admission.get('carried_policy_sha256')!=sha(Path(champ['code'])/'joint_safety.py'):raise AgentError('Counterexample policy binding mismatch')
 for m in champ['metrics'].values():
  for key in ('report','summary','contract'):verify(m[key])
  contract=read(m['contract']['path']);report=read(m['report']['path'])
  if contract.get('scene')!=case['scene']:raise AgentError('Counterexample borrowed another scene baseline')
  if contract.get('code_pins')!=champ['code_pins'] or not contract.get('fresh_history') or contract.get('score_kind')!='executed_path':raise AgentError('Counterexample baseline controller/history is not matched')
  if report.get('validation_contract')!=m['contract'] or not report.get('real_closed_loop') or not report.get('model_weights_unchanged') or len(report.get('trace',[]))!=m['controls'] or len(report.get('observations',[]))!=m['observations']:raise AgentError('Counterexample baseline does not match actual execution')
 adapter=load_adapter(task,config,state)
 if not adapter.verify_task_admission(task,activation):raise AgentError('Scene adapter rejected counterexample baseline')
 policy=copy.deepcopy(config['convergence']);policy.update(target_case=activation['target_case'],allow_control_failure_baseline=True)
 ledger=convergence.initial(task['id'],champ['metrics'],policy,state['simulation_only'])
 if config.get('host_governance',{}).get('enabled'):
  ledger['recovery_baseline']=True
  state['counterexample_training_task_id']=task['id']
 old=state['active_task_id']
 if old not in [x['task_id'] for x in state.get('task_archive',[])]:raise AgentError('Previous converged task must remain a regression target')
 proof=dict(schema='released_counterexample_task_admission_v1',release=release,task=task,activation=activation,previous_task=old,budget_reset=False,time=utc())
 file=Path(root)/'task_transitions'/(old+'_to_'+task['id']+'.json');write(file,proof)
 state.update(active_task_id=task['id'],active_task_spec=task,current_task_descriptor=copy.deepcopy(task),champion=champ,reference_metrics=activation['reference_metrics'],task_switch_pending=False,status='ready',diagnostic_cache={},task_case_ids=list(champ['metrics']),counterexample_task_admission=asset(file),generalization_pending=False,final_patch_status='provisional',registered_generalization_validated=False)
 state.pop('generalization',None);state.pop('accepted_patch',None)
 if activation.get('registry') is None:state.pop('registry',None)
 else:state['registry']=activation['registry']
 state['task_convergence'][task['id']]=ledger
 state.setdefault('training_scene_history',[])
 if case['scene'] not in state['training_scene_history']:state['training_scene_history'].append(case['scene'])
 index=next(i for i,x in enumerate(released['failed_cases']) if x['case']==case)
 from core import digest
 state.setdefault('counterexample_promotions',{})[digest(dict(release=release,index=index))]=asset(file)
 write(Path(root)/'working_champion.json',champ)
