"""Bounded campaign coordinator: evidence -> model -> patch -> tests -> paired GPU."""

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

import copy,json,os,time,traceback
pass  # Path is provided by the portable boundary above
from core import AgentError,BudgetExceeded,Deferred,LOCAL,ROOT,asset,code_pins,digest,event,identity,lock,pins,read,remaining,sha,utc,verify,verify_pins,write
from contracts import decision,normalize_response
from provider import HTTPProvider,TransientAPI,CompletedResponse
from evidence import collect,diagnostics
from validation import acceptance
import governance
import task_rotation
import generalization
import jobs
import search_governance as search
from final_verdict import finalize as final_qualification
from trajectory_quality import for_metrics as motion_quality
from adapter import HUGSIMAdapter

class Engine:
 def __init__(self,root,config=None,provider=None,adapter=None):
  self.root=Path(root).resolve();self.adapter=adapter or HUGSIMAdapter();self.config=config
  if self.config is None:self.config=read(self.root/'config.json')
  self.provider=provider or HTTPProvider(self.config['api'])
  saved=read(self.root/'state.json') if (self.root/'state.json').exists() else {}
  if adapter is None and saved.get('active_task_spec'):
   task=saved['active_task_spec']
   if task.get('adapter'):self.adapter=task_rotation.load_adapter(task,self.config,saved)
   else:
    restore=getattr(self.adapter,'restore_task',None)
    if restore is None:raise AgentError('Active task requires its registered adapter to restore; no scene fallback is allowed')
    restore(task,saved['champion'])
 def initialize(self,campaign):
  identity(campaign)
  if len(campaign)>48:raise AgentError('Campaign name must be at most 48 characters')
  self.root.mkdir(parents=True,exist_ok=True)
  if (self.root/'state.json').exists():raise AgentError('Campaign exists; use resume')
  inherited={}
  if search.enabled(self.config) and not self.adapter.simulation_only:
   if self.config.get('generalization',{}).get('mode')!='counterexample_cross_scene':raise AgentError('Real v9 requires a host-admitted fresh independent counterexample validation catalog; legacy public fixtures are not a validation pool')
   from budget_lineage import inherit
   inherit(inherited,self.config)
  champion=self.adapter.bootstrap(self.config)
  write(self.root/'config.json',self.config);write(self.root/'host_freeze.json',dict(code=pins(LOCAL),config_sha256=sha(self.root/'config.json')))
  generalization.policy(self.config)
  state=dict(final_patch_status='provisional',registered_generalization_validated=False,schema='autonomous_patch_campaign_v1',campaign=campaign,status='ready',started_epoch=None,updated_at=utc(),rounds_started=0,active_round=None,history=[],champion=champion,reference_metrics=champion['reference_metrics'],usage=dict(api_requests=0,charged_tokens=0,reported_tokens=0,gpu_dispatches=0),simulation_only=bool(self.adapter.simulation_only),production_driver_changed=False,global_patch_validated=False)
  state.update(inherited)
  if hasattr(self.adapter,'registered_pool'):
   state['registry']=self.adapter.registered_pool(self.config,champion)
   for key,item in state['registry'].items():
    if item['code']==champion['code']:state['champion']['pool_id']=key;state['champion']['profile']=item['profile']
   write(self.root/'candidate_registry.json',state['registry'])
  task_rotation.initialize(state,self.config)
  self.save(state);event(self.root,'initialized',version=champion['version'],simulation_only=state['simulation_only']);return state
 def save(self,state):
  state['updated_at']=utc();write(self.root/'state.json',state)
  r=state.get('active_round') or {};rid=r.get('id') or (state.get('history') or [{}])[-1].get('round');u=state['usage']
  lines=['# 补丁最新进展','', '状态：'+state['status'],'当前轮次：'+str(rid or '无'),'当前阶段：'+str(r.get('phase') or '已结束'),'','API已报告tokens：'+str(u['reported_tokens']),'保守计入tokens：'+str(u['charged_tokens']),'GPU派发次数：'+str(u['gpu_dispatches']),'']
  if rid:
   folder=self.root/'rounds'/rid
   for name,label in [('failure_reason.json','失败分析'),('patch_instruction.json','补丁方案'),('acceptance.json','验收结果'),('round_result.json','本轮结果')]:
    path=folder/name;lines.append('['+label+']('+str(path)+')' if path.exists() else label+'：尚未生成')
  if state.get('active_task_id'):
   ledger=state['task_convergence'][state['active_task_id']];lines+=['','当前任务：'+state['active_task_id'],'连续小增幅：'+str(ledger['small_gain_streak'])+'/'+str(ledger['policy']['window']),'收敛阈值：'+str(ledger['policy']['minimum_gain'])]
  if search.enabled(self.config) and state.get('active_search_stage'):
   stage=state['search_stages'][state['active_search_stage']];lines+=['','搜索阶段：'+stage['phase'],'合格最好分：'+str(stage['best_qualified_score']),'有效评估：'+str(stage['effective_evaluations'])+'/'+str(self.config['host_governance']['max_effective_evaluations_per_stage']),'搜索Patience：'+str(stage['search_stagnation'])+'/'+str(self.config['host_governance']['search_patience'])]
  if state.get('task_wait_reason'):lines+=['','任务等待：'+state['task_wait_reason'],'[下一任务所需资料]('+str(self.root/'next_task_request.json')+')']
  lines+=['','## 已结束轮次','']+[row['round']+'：'+row['status']+' / '+row['reason'] for row in state.get('history',[])]
  # A presentation-file failure must not abort a saved engineering state.
  try:
   tmp=self.root/'.LATEST.tmp';tmp.write_text('\n'.join(lines)+'\n');os.replace(tmp,self.root/'LATEST.md')
  except OSError:pass

 def verify_host(self):
  freeze=read(self.root/'host_freeze.json');verify_pins(freeze['code'])
  saved=read(self.root/'state.json');spec=saved.get('active_task_spec')
  if spec and spec.get('adapter'):
   if spec not in self.config['task_queue']['tasks']:
    from task_rotation import verify_counterexample_task
    verify_counterexample_task(saved,self.config)
   verify(spec['adapter']['module'])
  if freeze['config_sha256']!=sha(self.root/'config.json'):raise AgentError('Frozen campaign configuration changed')
 def stop_requested(self):return (self.root/'STOP').exists() or (ROOT/'outputs/gpu_test_service/STOP').exists()
 def round_dir(self,r):return self.root/'rounds'/r['id']
 def new_round(self,state):
  if search.enabled(self.config):
   stage=search.before_round(state,self.config)
   if not stage.get('resource_admission') and not state['simulation_only']:
    profile=getattr(self.adapter,'resource_profile',lambda:{})()
    plan=search.admission(state,self.config,profile);write(self.root/'stage_resource_admission.json',plan)
    if not plan['passed']:raise BudgetExceeded('Insufficient shared resources for minimum stage and validation; no automatic allocation')
    stage['resource_admission']=plan
  if state['rounds_started']>=self.config['budgets']['max_rounds']:raise BudgetExceeded('Round budget exhausted')
  state['rounds_started']+=1;rid=state['campaign']+'_r'+str(state['rounds_started']).zfill(3);identity(rid)
  selected=governance.select_parent(state['registry'],state['champion'],state['reference_metrics']) if 'registry' in state else copy.deepcopy(state['champion'])
  r=dict(task_id=state.get('active_task_id'),id=rid,phase='capture',parent=selected,model_attempts=0,diagnostic_turns=0,stage_index=0,attempt_number=1,job_retry=0,status='active')
  self.round_dir(r).mkdir(parents=True,exist_ok=False);state['active_round']=r;self.save(state);event(self.root,'round_started',round=rid,parent=r['parent']['version']);return r
 def finish(self,state,r,status,reason,**details):
  if r.get('attempt'):details.setdefault('matrix',r['attempt']['output'])
  decision_file=self.round_dir(r)/'decision.json'
  if decision_file.exists():
   d=read(decision_file)['patch_instruction'];details['patch_summary']=d['summary'];details['patch_rationale']=d['rationale'];details['patch_operations']=[dict(path=o['path'],before=o['before'][:3000],after=o['after'][:3000]) for o in d['operations']]
  failure=r.get('last_job_failure',{})
  if failure.get('log'):details['failure_log_tail']=Path(failure['log']['path']).read_text(errors='replace')[-5000:]
  r.update(status=status,reason=reason,finished_at=utc());write(self.round_dir(r)/'round_result.json',{**r,**details})
  state['history'].append(dict(round=r['id'],status=status,reason=reason,**details));state['active_round']=None
  self.save(state);event(self.root,'round_finished',round=r['id'],status=status,reason=reason)
 def charge(self,state,reservation):
  remaining(state,self.config);u=state['usage'];b=self.config['budgets']
  if u['api_requests']+1>b['max_api_requests'] or u['charged_tokens']+reservation>b['max_api_tokens']:raise BudgetExceeded('API request/token budget exhausted')
  u['api_requests']+=1;u['charged_tokens']+=reservation;self.save(state)
 def reconcile(self,state,meta):
  u=state['usage'];known=meta.get('usage_known',True)
  if known:
   u['reported_tokens']+=meta['usage_tokens']
   u['charged_tokens']+=meta['usage_tokens']-meta['reserved_tokens']
   for field in ('prompt_tokens','completion_tokens'):
    u[field]=u.get(field,0)+meta.get('usage',{}).get(field,0)
  if u['charged_tokens']>self.config['budgets']['max_api_tokens']:raise BudgetExceeded('Reported API token budget exceeded')
 def model_step(self,state,r):
  if r['model_attempts']>=self.config['budgets']['max_model_attempts_per_round']:
   self.finish(state,r,'rejected','model_attempt_budget_exhausted',feedback=r.get('feedback'));return True
  rd=self.round_dir(r);missing=self.provider.readiness()
  if missing:state.update(status='waiting_api',missing_api_variables=missing);self.save(state);return False
  evidence=read(r['evidence']['path']);verify(r['evidence'])
  evidence['diagnostic_runtime']=dict(remaining=max(0,self.config['budgets']['max_diagnostics_per_round']-r['diagnostic_turns']),rejected_requests=r.get('diagnostic_rejections',0))
  # Bind the model-visible runtime data to a new immutable evidence asset.
  runtime_path=rd/('evidence_model_'+str(r['model_attempts']+1).zfill(3)+'.json');write(runtime_path,evidence);r['evidence']=asset(runtime_path)
  previous_response=None
  if r.get('last_rejected_response'):
   verify(r['last_rejected_response']);prior=read(r['last_rejected_response']['path']).get('content');encoded=json.dumps(prior,ensure_ascii=False)
   if len(encoded)<=60000:previous_response=prior
  schema=read(LOCAL/'schemas/decision.schema.json')
  ctx=evidence.get('iteration_context')
  case_ids=evidence.get('task_contract',{}).get('case_ids')
  if case_ids:
   cases=schema['properties']['edit_instruction']['properties']['cases'];cases['items']['enum']=case_ids;cases['minItems']=len(case_ids);cases['maxItems']=len(case_ids)
  if ctx:
   spec=schema['properties']['iteration_plan']['properties'];spec['base_id']['enum']=ctx['frontier_ids'];spec['compared_versions']['items']['enum']=ctx['allowed_comparison_ids'];spec['compared_versions']['uniqueItems']=True
  messages=[{'role':'system','content':(LOCAL/'prompt.txt').read_text()},{'role':'user','content':json.dumps(dict(evidence=evidence,evidence_sha256=digest(evidence),validation_feedback=r.get('feedback'),previous_response=previous_response,required_response_schema=schema),ensure_ascii=False)}]
  record=rd/'model_calls'/('request_'+str(r['model_attempts']+1).zfill(3)+'.json')
  r['model_attempts']+=1;self.save(state)
  try:
   # Bound a request by the remaining campaign wall budget as well.
   if isinstance(self.provider,HTTPProvider):self.provider.config=dict(self.config['api'],timeout_seconds=min(self.config['api']['timeout_seconds'],max(1.,remaining(state,self.config))))
   write(record.with_suffix('.request.json'),dict(time=utc(),messages=messages,evidence=r['evidence'],response_schema=asset(LOCAL/'schemas/decision.schema.json'),resolved_response_schema=schema,credentials_persisted=False))
   raw,meta=self.provider.complete(messages,schema,lambda n:self.charge(state,n),record)
   self.reconcile(state,meta)
   r['transport_failures']=0
   normalized,audit=normalize_response(raw,evidence)
   proposal=decision(normalized,evidence)
   write(rd/('normalization_'+str(r['model_attempts']).zfill(3)+'.json'),dict(changes=audit,source=asset(record)))
   if proposal['decision']=='patch' and hasattr(self.adapter,'preflight_proposal'):self.adapter.preflight_proposal(r['parent'],proposal,self.config)
   r.pop('last_rejected_response',None)
  except CompletedResponse as e:
   self.reconcile(state,e.metadata)
   r['feedback']=dict(type='incomplete_api_response',message=str(e),metadata=e.metadata)
   state.update(status='blocked_api_response',reason=str(e));self.save(state)
   event(self.root,'api_response_blocked',round=r['id'],reason=str(e));return False
  except TransientAPI as e:
   r['transport_failures']=r.get('transport_failures',0)+1
   if r['transport_failures']>=self.config['api'].get('max_consecutive_transport_failures',2):
    state.update(status='blocked_api_transport',reason=str(e));self.save(state);return False
   r['retry_after']=time.time()+self.config['budgets']['retry_delay_seconds'];state.update(status='waiting_api_retry',reason=str(e));self.save(state);return True
  except Deferred:
   self.save(state);raise
  except BudgetExceeded:raise
  except (AgentError,ValueError,KeyError,IndexError,TypeError) as e:
   if record.exists() and 'content' in read(record):r['last_rejected_response']=asset(record)
   r['feedback']=dict(type='invalid_model_response',message=str(e)[:1500]);write(rd/'validation_feedback_latest.json',r['feedback']);self.save(state)
   event(self.root,'model_response_rejected',round=r['id'],error=str(e)[:1500])
   if r['model_attempts']>=self.config['budgets']['max_model_attempts_per_round']:
    self.finish(state,r,'rejected','invalid_model_decision',feedback=r['feedback'])
   return True
  write(rd/('decision_'+str(r['model_attempts']).zfill(3)+'.json'),dict(decision=proposal,provider=meta,evidence=r['evidence']))
  for name in ('failure_reason','edit_instruction','patch_instruction'):write(rd/(name+'.json'),dict(schema=name+'_autonomous_v1',decision=proposal['decision'],author=meta,evidence=r['evidence'],content=proposal[name]))
  if proposal['decision']=='patch' and 'registry' in state:
   plan=proposal['iteration_plan'];write(rd/('base_selection_'+str(r['model_attempts'])+'.json'),dict(plan=plan,comparators=evidence['iteration_context'],evidence=r['evidence']))
   selected=plan['base_id']
   if selected!=r['parent'].get('pool_id'):
    r.update(parent=copy.deepcopy(state['registry'][selected]),phase='capture',feedback={'type':'base_switched','message':'Re-author code against newly supplied source; no old patch was applied'})
    self.save(state);return True
   r['base_selection']=plan
  if proposal['decision']=='stop':
   reviewer=getattr(self.adapter,'review_stop',None)
   review=reviewer(proposal,evidence) if reviewer else None
   if review and not r.get('stop_review_applied') and r['model_attempts']<self.config['budgets']['max_model_attempts_per_round']:
    r['stop_review_applied']=True;r['feedback']=review;self.save(state);return True
   self.finish(state,r,'no_supported_patch','model_stopped',analysis=proposal['failure_reason']);state['status']='model_stopped';self.save(state);return False
  if proposal['decision']=='diagnose':
   if r['diagnostic_turns']>=self.config['budgets']['max_diagnostics_per_round']:
    r['feedback']=dict(type='diagnostic_limit_reached',message='Successful diagnostic budget exhausted. Use delivered evidence for patch or stop; do not request more diagnostics.')
    self.save(state)
    if r['model_attempts']>=self.config['budgets']['max_model_attempts_per_round']:self.finish(state,r,'rejected','diagnostic_budget_exhausted')
    return True
   try:new,ids=diagnostics(evidence,proposal['diagnostics'],self.config['context']['max_chars'])
   except AgentError as e:
    r['last_rejected_response']=asset(record)
    r['feedback']=dict(type='diagnostic_request_rejected',message=str(e));r['diagnostic_rejections']=r.get('diagnostic_rejections',0)+1;self.save(state)
    if r['model_attempts']>=self.config['budgets']['max_model_attempts_per_round']:self.finish(state,r,'rejected','diagnostic_request_rejected',feedback=r['feedback'])
    return True
   fresh=any(i not in evidence['items'] for i in ids)
   state.setdefault('diagnostic_cache',{}).update({i:new['items'][i] for i in ids})
   r['diagnostic_turns']+=int(fresh);path=rd/('evidence_'+str(r['diagnostic_turns']).zfill(2)+'.json');write(path,new);r['evidence']=asset(path);r['feedback']=dict(type='diagnostic_result',new_evidence_ids=ids);self.save(state);return True
  for name in ('failure_reason','edit_instruction','patch_instruction'):write(rd/(name+'.json'),dict(schema=name+'_autonomous_v1',author=meta,evidence=r['evidence'],content=proposal[name]))
  write(rd/'decision.json',proposal);r['round_artifacts']=[asset(rd/(n+'.json')) for n in ('failure_reason','edit_instruction','patch_instruction')]
  r.update(phase='apply',decision=asset(rd/'decision.json'));self.save(state);return True
 def apply_step(self,state,r):
  rd=self.round_dir(r);verify(r['decision']);p=read(r['decision']['path'])
  try:
   bundle=self.adapter.build(rd,r['id'],r['parent'],p,self.config);r['bundle']=bundle;r['attempt']=self.adapter.attempt(rd,bundle,r['attempt_number']);r.update(phase='validate',stage_index=0);self.save(state)
  except (AgentError,SyntaxError,ValueError,FileExistsError) as e:self.finish(state,r,'rejected','patch_application_failed',error=str(e)[:2000])
 def validate_step(self,state,r):
  rd=self.round_dir(r)
  verify(r['evidence']);verify(r['decision'])
  for item in r.get('round_artifacts',[]):verify(item)
  stage_config=copy.deepcopy(self.config);stage_config['_released_counterexamples']=state.get('released_counterexample_manifests',[])
  stages=self.adapter.stages(rd,r['bundle'],r['attempt'],stage_config)
  if search.enabled(self.config) and not state['simulation_only']:
   from offline_motion_gate import insert
   stages=insert(stages,rd,r['bundle'],stage_config)
  if r['stage_index']>=len(stages):r['phase']='evaluate';self.save(state);return
  stage=stages[r['stage_index']];tag=stage['name']+'_a'+str(r['attempt_number'])+'_j'+str(r['job_retry']);folder=rd/'jobs'/tag
  inputs={**read(self.root/'host_freeze.json')['code'],**stage['inputs']}
  active_spec=state.get('active_task_spec',{})
  if active_spec.get('adapter'):
   a=active_spec['adapter']['module'];inputs[a['path']]=a['sha256']
  for item in [r['evidence'],r['decision'],*r.get('round_artifacts',[])]:inputs[item['path']]=item['sha256']
  for p in stage.get('deps',[]):inputs[str(Path(p).resolve())]=sha(p)
  timeout=min(stage['timeout'],remaining(state,self.config))
  if stage['gpu'] and not (folder/'spec.json').exists():
   if state['usage'].get('gpu_attempts',0)>=self.config['budgets']['max_gpu_attempts']:raise BudgetExceeded('GPU dispatch budget exhausted')
   state['usage']['gpu_dispatches']+=1;state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;self.save(state)
  receipt=jobs.launch(folder,stage['command'],ROOT,timeout,inputs,stage['outputs'],self.root/'STOP')
  state.update(status='running_'+stage['name']);r['current_job']=str(folder);self.save(state)
  if receipt['status']=='running':return
  if receipt['status']!='succeeded':
   kind=self.adapter.classify_failure(stage,receipt);r['last_job_failure']=dict(stage=stage['name'],kind=kind,receipt=str(folder/'receipt.json'),log=receipt.get('log'),status=receipt['status'])
   event(self.root,'stage_failed',round=r['id'],stage=stage['name'],failure_kind=kind)
   if kind in ('gpu_busy','gpu_unavailable'):
    state['usage']['gpu_attempts']-=1
    r['job_retry']+=1;state['status']='waiting_gpu';r['retry_after']=time.time()+self.config['budgets']['retry_delay_seconds'];self.save(state);return
   if kind in ('interrupted','infrastructure'):
    if search.enabled(self.config) and not search.infrastructure_failure(state,self.config,str(folder)):
     state['status']='infrastructure_retry_quota_exhausted';self.save(state);return
    if not stage['gpu']:
     r['offline_retries']=r.get('offline_retries',0)+1
     if r['offline_retries']<=self.config['budgets']['max_infrastructure_attempts']:
      r['job_retry']+=1;self.save(state);return
     self.finish(state,r,'rejected','offline_task_interrupted_repeatedly',failure=r['last_job_failure']);return
    if r['attempt_number']<self.config['budgets']['max_infrastructure_attempts']:
     r['attempt_number']+=1;r['attempt']=self.adapter.attempt(rd,r['bundle'],r['attempt_number']);r['job_retry']=0
     # All candidate/causal evidence remains pinned. A new harness and output
     # preserve every partial episode; no neural-history splice or replay.
     new_stages=self.adapter.stages(rd,r['bundle'],r['attempt'],stage_config)
     r['stage_index']=next(i for i,s in enumerate(new_stages) if s['name']=='harness_cpu');self.save(state);return
    state['status']='blocked_infrastructure';r['status']='infrastructure_unresolved';self.save(state);return
   self.finish(state,r,'rejected','offline_validation_failed',failure=r['last_job_failure']);return
  try:
   if stage['name']=='host_motion_contract':
    proof=read(rd/'checks/host_motion_contract.json')
    if not proof['passed'] or proof['tests']<3 or proof['controller']!=asset(Path(r['bundle']['code'])/'joint_safety.py') or any(proof[k] for k in ['failures','errors','skipped','expected_failures','unexpected_successes']):raise AgentError('Host kinematic contract failed before GPU')
   else:self.adapter.validate_stage(stage['name'],rd,r['bundle'],r['attempt'])
  except (AgentError,ValueError,KeyError,TypeError) as e:
   self.finish(state,r,'rejected','offline_proof_rejected',stage=stage['name'],error=str(e)[:2000]);return
  r['stage_index']+=1;r['job_retry']=0;r.pop('current_job',None);self.save(state);event(self.root,'stage_passed',round=r['id'],stage=stage['name'],receipt=asset(folder/'receipt.json'))
 def evaluate_step(self,state,r):
  rd=self.round_dir(r)
  try:
   metrics,strict=self.adapter.results(r['bundle'],r['attempt'])
   judge=getattr(self.adapter,'acceptance',acceptance)
   verdict=judge(metrics,state['champion']['metrics'],state['reference_metrics'],self.config['acceptance'],strict)
   if self.config.get('generalization',{}).get('mode') in ('blind_cross_scene','counterexample_cross_scene'):
    from blind_validation import training_verdict
    verdict=training_verdict(verdict,state,metrics,self.config['acceptance'])
   r['convergence_best_before']=state['champion']['metrics'][state.get('task_convergence',{}).get(state.get('active_task_id'),{}).get('policy',self.config.get('convergence',{})).get('target_case','cutin551')]['score']['hdscore']
   retained=True
   if state.get('task_archive'):
    retained=task_rotation.verify_retention(self.adapter,r['bundle'],r['attempt'],state['task_archive'],self.config['acceptance'],state['simulation_only'],self.config)
    if not retained:verdict['accepted']=False;verdict['reasons'].append('previous_task_nonregression_unverified')
   if 'registry' in state:
    verdict=governance.retain_gains(verdict,metrics,r['parent'],state['registry'])
    r['convergence_best_before']=verdict['gain_retention']['previous_best_hdscore']
    if search.enabled(self.config):
     verdict=final_qualification(verdict,metrics,state['champion']['metrics'],self.config['acceptance']['numeric_tolerance'],retained,motion_quality(metrics,self.config,state['simulation_only']),r['bundle']['candidate_pins'])
    governance.add_result(state['registry'],r['parent'],r['bundle'],r['attempt']['output'],metrics,verdict if search.enabled(self.config) else None)
    if 'previous_task_nonregression_unverified' in verdict['reasons']:state['registry'][r['bundle']['round_id']]['eligible']=False
    write(self.root/'candidate_registry.json',state['registry'])
  except (AgentError,ValueError,KeyError,TypeError) as e:
   self.finish(state,r,'rejected','invalid_evaluation',error=str(e)[:2000]);return
  if search.enabled(self.config) and 'registry' not in state:
   verdict=final_qualification(verdict,metrics,state['champion']['metrics'],self.config['acceptance']['numeric_tolerance'],retained,motion_quality(metrics,self.config,state['simulation_only']),r['bundle']['candidate_pins'])
  verdict.update(simulation_only=state['simulation_only'],GPU_executed=not state['simulation_only'],provisional=True,confirmed_patch=False)
  write(rd/'acceptance.json',dict(**verdict,metrics=metrics,strict_evaluation=asset(Path(r['attempt']['output'])/'evaluation.json')))
  r.update(verdict=verdict,metrics=metrics,phase='presentation');self.save(state)
  if not self.config.get('presentation',{}).get('enabled',False) or state['simulation_only']:self.finalize_evaluation(state,r)

 def finalize_evaluation(self,state,r):
  rd=self.round_dir(r);verdict=r['verdict'];metrics=r['metrics']
  if search.enabled(self.config):
   search.ensure(state,self.config)
   candidate=dict(version=r['id'],code=r['bundle']['code'],code_pins=r['bundle']['candidate_pins'],metrics=metrics)
   search.observe(state,self.config,r['id'],candidate,metrics,verdict)
   write(rd/'search_governance.json',state['search_stages'][state['active_search_stage']])
  if verdict['accepted']:
   old=copy.deepcopy(state['champion']);b=r['bundle']
   champion=dict(version=r['id'],code=b['code'],release=b['release'],release_name=b['release_name'],code_pins=b['candidate_pins'],matrix=r['attempt']['output'],metrics=metrics,reference_metrics=state['reference_metrics'],acceptance=asset(rd/'acceptance.json'))
   # Write the ledger and pointer before advancing to another model decision.
   write(rd/'promotion.json',dict(previous=old,candidate=champion,acceptance=asset(rd/'acceptance.json'),scope='provisional local working champion; not a confirmed patch',production_driver_changed=False,global_patch_validated=False))
   if 'registry' in state:champion.update(pool_id=r['id'],profile=r['parent']['profile'])
   state['champion']=champion;state.update(final_patch_status='provisional',registered_generalization_validated=False);write(self.root/'working_champion.json',champion)
  if self.config.get('convergence',{}).get('enabled'):task_rotation.record(state,r['id'],metrics,verdict,r['convergence_best_before'],self.root)
  self.finish(state,r,'accepted' if verdict['accepted'] else 'tradeoff_retained' if verdict.get('gain_retention') and 'lost_high_score_branch_gain' in verdict['reasons'] else 'rejected','paired_acceptance_passed' if verdict['accepted'] else 'paired_acceptance_failed',acceptance=verdict,media=r.get('media'))

 def presentation_step(self,state,r):
  if state.get('task_case_ids') and state['task_case_ids']!=['original551','cutin551']:
   r['media']={'status':'not_registered_for_active_task'};self.finalize_evaluation(state,r);return
  rd=self.round_dir(r);folder=rd/'jobs/presentation';out=Path(r['attempt']['output']);baseline=Path(r['parent']['matrix'])/'cases/cutin551/on/B';candidate=out/'cases/cutin551/on/B'
  inputs={**read(self.root/'host_freeze.json')['code'],**r['bundle']['candidate_pins']}
  for p in (baseline/'run_report.json',candidate/'run_report.json'):inputs[str(p.resolve())]=sha(p)
  command=[str(ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'),'-u','-B',str(LOCAL/'presentation.py'),'--code',r['bundle']['code'],'--output',str(out),'--baseline',str(baseline),'--candidate',str(candidate)]
  result=jobs.launch(folder,command,ROOT,min(self.config['presentation']['timeout_seconds'],remaining(state,self.config)),inputs,[out/'presentation_Boff_Bon/artifacts.json'],self.root/'STOP')
  r['current_job']=str(folder);state['status']='rendering_comparison';self.save(state)
  if result['status']=='running':return
  r['media']={'status':result['status'],'receipt':str(folder/'receipt.json')}
  if result['status']=='succeeded':
   report=read(out/'presentation_Boff_Bon/artifacts.json')
   for a in (report['video'],report['plot'],*report['source_reports']):verify(a)
   if not report['full_decode_passed']:raise AgentError('Comparison video did not decode')
   r['media']['artifacts']=asset(out/'presentation_Boff_Bon/artifacts.json')
  self.finalize_evaluation(state,r)

 def tick(self,state):
  self.verify_host();remaining(state,self.config)
  if self.stop_requested():
   if state.get('counterexample_preparation_job'):jobs.cancel(Path(state['counterexample_preparation_job']))
   if (state.get('generalization') or {}).get('current_job'):jobs.cancel(Path(state['generalization']['current_job']))
   if (state.get('active_round') or {}).get('current_job'):jobs.cancel(Path(state['active_round']['current_job']))
   state['status']='stopped';self.save(state);return False
  verify_pins(state['champion']['code_pins'])
  if search.enabled(self.config) and state.get('active_task_id') and not state.get('active_round') and not state.get('task_switch_pending'):
   stage=search.ensure(state,self.config)
   if stage['search_stopped']:
    state['status']=stage['stop_reason'];self.save(state);return False
  if (state.get('blind_repair') or {}).get('audit_required'):
   reviewer=getattr(self.adapter,'training_review',None)
   if reviewer is None:state['status']='waiting_training_review_adapter';self.save(state);return False
   reviewed=reviewer(state,self.config,self.root);self.save(state)
   if reviewed is not True:return reviewed is None
  if state.get('task_switch_pending'):
   gate=generalization.step(state,self.config,self.adapter,self.root);self.save(state)
   if state.pop('adapter_switch_pending',False):
    self.adapter=task_rotation.load_adapter(state['active_task_spec'],self.config,state);self.save(state)
    return True
   if gate is not True:return gate is None
   more,self.adapter=task_rotation.advance(state,self.config,self.adapter,self.root);self.save(state);return more
  r=state['active_round'] or self.new_round(state)
  if r.get('retry_after',0)>time.time():return True
  r.pop('retry_after',None)
  if r['phase']=='capture':
   latest=state['history'][-1].get('matrix') if state['history'] else None
   verify_pins(r['parent']['code_pins'])
   cfg=copy.deepcopy(self.config)
   cfg['_released_counterexamples']=state.get('released_counterexample_manifests',[])
   if state.get('public_training_audit'):cfg['_public_training_audit']=state['public_training_audit']
   if 'registry' in state:cfg['evidence_roots']+=sorted({v['matrix'] for v in state['registry'].values()})
   collector=getattr(self.adapter,'collect_evidence',collect)
   e=collector(cfg,state['champion'],state.get('prior_campaign_history',[])+state['history'],latest)
   if search.enabled(self.config):
    stage=search.ensure(state,self.config);e['qualified_search']={k:copy.deepcopy(stage[k]) for k in ['phase','best_qualified_candidate','best_qualified_score','effective_evaluations','search_stagnation']}
    e['qualified_search']['last_attempt']=(state['history'][-1] if state['history'] else None)
   scope=getattr(self.adapter,'prepare_repair_scope',None)
   if scope:e=scope(state,r,e,self.root)
   if 'registry' in state:
    parent=r['parent'];e['selected_parent_profile']=parent['profile'];e['selected_parent_id']=parent['pool_id'];source=Path(parent['code'])/'joint_safety.py';e['items']['source'].update(asset=asset(source),text=source.read_text())
    e=governance.attach(e,state['registry'],state['champion'],state['reference_metrics'])
    from evidence import bounded
    e=bounded(e,self.config['context']['max_chars'])
   if state.get('task_case_ids'):e['task_contract']={'case_ids':state['task_case_ids']}
   if state.get('active_task_id'):
    ledger=state['task_convergence'][state['active_task_id']];e['task_progress']={'task_id':state['active_task_id'],'small_gain_streak':ledger['small_gain_streak'],'window':ledger['policy']['window'],'minimum_gain':ledger['policy']['minimum_gain']}
   from evidence import decorate,restore_diagnostics
   e=decorate(e)
   e=restore_diagnostics(e,state.get('diagnostic_cache',{}),self.config['context']['max_chars'])
   if self.config.get('generalization',{}).get('mode') in ('blind_cross_scene','counterexample_cross_scene'):
    from privacy import public_evidence
    e=public_evidence(e,state,self.config,self.root)
   count=r.get('capture_count',0);p=self.round_dir(r)/('evidence_capture_'+str(count).zfill(3)+'.json');r['capture_count']=count+1
   write(p,e);r.update(evidence=asset(p),phase='propose');self.save(state)
  elif r['phase']=='propose':return self.model_step(state,r)
  elif r['phase']=='apply':self.apply_step(state,r)
  elif r['phase']=='validate':self.validate_step(state,r)
  elif r['phase']=='evaluate':self.evaluate_step(state,r)
  elif r['phase']=='presentation':self.presentation_step(state,r)
  else:raise AgentError('Unknown durable round phase')
  return state['status']!='blocked_infrastructure'
 def run(self,one_step=False):
  with lock(self.root/'coordinator.lock'):
   state=read(self.root/'state.json')
   if state['started_epoch'] is None and not self.provider.readiness():state['started_epoch']=time.time();self.save(state)
   if state['status'] in ('model_stopped','budget_exhausted','blocked_infrastructure','stopped','blocked_api_response','blocked_api_transport'):return state
   try:
    while self.tick(state):
     if one_step:break
     time.sleep(self.config['budgets']['poll_seconds'])
   except BudgetExceeded as e:
    if state.get('counterexample_preparation_job'):jobs.cancel(Path(state['counterexample_preparation_job']))
    if self.config.get('generalization',{}).get('mode') in ('blind_cross_scene','counterexample_cross_scene'):
     from blind_validation import cancel
     cancel(self.root)
    if state.get('public_training_audit_job'):jobs.cancel(Path(state['public_training_audit_job']))
    if (state.get('generalization') or {}).get('current_job'):jobs.cancel(Path(state['generalization']['current_job']))
    active=state.get('active_round')
    if (active or {}).get('current_job'):jobs.cancel(Path(active['current_job']))
    if active and active.get('phase')=='presentation' and active.get('verdict'):
     active['media']={'status':'budget_exhausted'};self.finalize_evaluation(state,active)
    state.update(status='budget_exhausted',reason=str(e));write(self.root/'STOP',dict(reason=str(e),time=utc()));self.save(state);event(self.root,'budget_exhausted',reason=str(e))
   except Deferred as e:
    state.update(status='waiting_api' if r'API' in str(e) else 'deferred',reason=str(e));self.save(state)
   except KeyboardInterrupt:
    state.update(status='coordinator_detached',reason='Registered child job retained; resume to observe, stop to cancel');self.save(state)
   except (AgentError,ValueError,KeyError,OSError) as e:
    state.update(status='integrity_or_executor_error',reason=str(e)[:2000]);self.save(state);event(self.root,'executor_error',error_type=type(e).__name__,reason=str(e)[:2000])
   self.report(state);return state
 def report(self,state=None):
  state=state or read(self.root/'state.json');lines=['# 自动补丁运行记录','',f"状态：{state['status']}",f"当前工作版本：{state['champion']['version']}",f"模拟验证：{state['simulation_only']}",'生产入口未自动修改；跨场景有效性与正式认证未宣称。','', '|轮次|结果|原因|','|---|---|---|']
  for row in state['history']:lines.append('|'+row['round']+'|'+row['status']+'|'+row['reason']+'|')
  lines+=['', '预算使用：'+json.dumps(state['usage'],ensure_ascii=False),'', '恢复任务从完整阶段证据接续；部分闭环在新目录重跑，不拼接感知历史。']
  (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
