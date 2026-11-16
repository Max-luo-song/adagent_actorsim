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
from contracts import decision
from provider import HTTPProvider,TransientAPI,CompletedResponse
from evidence import collect,diagnostics
from validation import acceptance
import governance
import jobs
from adapter import HUGSIMAdapter

class Engine:
 def __init__(self,root,config=None,provider=None,adapter=None):
  self.root=Path(root).resolve();self.adapter=adapter or HUGSIMAdapter();self.config=config
  if self.config is None:self.config=read(self.root/'config.json')
  self.provider=provider or HTTPProvider(self.config['api'])
 def initialize(self,campaign):
  identity(campaign)
  if len(campaign)>48:raise AgentError('Campaign name must be at most 48 characters')
  self.root.mkdir(parents=True,exist_ok=True)
  if (self.root/'state.json').exists():raise AgentError('Campaign exists; use resume')
  champion=self.adapter.bootstrap(self.config)
  write(self.root/'config.json',self.config);write(self.root/'host_freeze.json',dict(code=pins(LOCAL),config_sha256=sha(self.root/'config.json')))
  state=dict(schema='autonomous_patch_campaign_v1',campaign=campaign,status='ready',started_epoch=None,updated_at=utc(),rounds_started=0,active_round=None,history=[],champion=champion,reference_metrics=champion['reference_metrics'],usage=dict(api_requests=0,charged_tokens=0,reported_tokens=0,gpu_dispatches=0),simulation_only=bool(self.adapter.simulation_only),production_driver_changed=False,global_patch_validated=False)
  if hasattr(self.adapter,'registered_pool'):
   state['registry']=self.adapter.registered_pool(self.config,champion)
   for key,item in state['registry'].items():
    if item['code']==champion['code']:state['champion']['pool_id']=key;state['champion']['profile']=item['profile']
   write(self.root/'candidate_registry.json',state['registry'])
  self.save(state);event(self.root,'initialized',version=champion['version'],simulation_only=state['simulation_only']);return state
 def save(self,state):state['updated_at']=utc();write(self.root/'state.json',state)
 def verify_host(self):
  freeze=read(self.root/'host_freeze.json');verify_pins(freeze['code'])
  if freeze['config_sha256']!=sha(self.root/'config.json'):raise AgentError('Frozen campaign configuration changed')
 def stop_requested(self):return (self.root/'STOP').exists() or (ROOT/'outputs/gpu_test_service/STOP').exists()
 def round_dir(self,r):return self.root/'rounds'/r['id']
 def new_round(self,state):
  if state['rounds_started']>=self.config['budgets']['max_rounds']:raise BudgetExceeded('Round budget exhausted')
  state['rounds_started']+=1;rid=state['campaign']+'_r'+str(state['rounds_started']).zfill(3);identity(rid)
  selected=governance.select_parent(state['registry'],state['champion'],state['reference_metrics']) if 'registry' in state else copy.deepcopy(state['champion'])
  r=dict(id=rid,phase='capture',parent=selected,model_attempts=0,diagnostic_turns=0,stage_index=0,attempt_number=1,job_retry=0,status='active')
  self.round_dir(r).mkdir(parents=True,exist_ok=False);state['active_round']=r;self.save(state);event(self.root,'round_started',round=rid,parent=r['parent']['version']);return r
 def finish(self,state,r,status,reason,**details):
  if r.get('attempt'):details.setdefault('matrix',r['attempt']['output'])
  decision_file=self.round_dir(r)/'decision.json'
  if decision_file.exists():
   d=read(decision_file)['patch_instruction'];details['patch_summary']=d['summary'];details['patch_rationale']=d['rationale'];details['patch_operations']=[dict(path=o['path'],before=o['before'][:3000],after=o['after'][:3000]) for o in d['operations']]
  failure=r.get('last_job_failure',{})
  if failure.get('log'):details['failure_log_tail']=Path(failure['log']['path']).read_text(errors='replace')[-5000:]
  r.update(status=status,reason=reason,finished_at=utc());write(self.round_dir(r)/'round_result.json',dict(**r,**details))
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
  rd=self.round_dir(r);missing=self.provider.readiness()
  if missing:state.update(status='waiting_api',missing_api_variables=missing);self.save(state);return False
  evidence=read(r['evidence']['path']);verify(r['evidence'])
  messages=[{'role':'system','content':(LOCAL/'prompt.txt').read_text()},{'role':'user','content':json.dumps(dict(evidence=evidence,evidence_sha256=digest(evidence),validation_feedback=r.get('feedback'),required_response_schema=read(LOCAL/'schemas/decision.schema.json')),ensure_ascii=False)}]
  record=rd/'model_calls'/('request_'+str(r['model_attempts']+1).zfill(3)+'.json')
  r['model_attempts']+=1;self.save(state)
  try:
   # Bound a request by the remaining campaign wall budget as well.
   if isinstance(self.provider,HTTPProvider):self.provider.config=dict(self.config['api'],timeout_seconds=min(self.config['api']['timeout_seconds'],max(1.,remaining(state,self.config))))
   write(record.with_suffix('.request.json'),dict(time=utc(),messages=messages,evidence=r['evidence'],response_schema=asset(LOCAL/'schemas/decision.schema.json'),credentials_persisted=False))
   raw,meta=self.provider.complete(messages,read(LOCAL/'schemas/decision.schema.json'),lambda n:self.charge(state,n),record)
   self.reconcile(state,meta)
   r['transport_failures']=0
   proposal=decision(raw,evidence)
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
   r['feedback']=dict(type='invalid_model_response',message=str(e)[:1500]);self.save(state)
   event(self.root,'model_response_rejected',round=r['id'],error=str(e)[:1500])
   if r['model_attempts']>=self.config['budgets']['max_model_attempts_per_round']:
    self.finish(state,r,'rejected','invalid_model_decision',feedback=r['feedback'])
   return True
  write(rd/('decision_'+str(r['model_attempts']).zfill(3)+'.json'),dict(decision=proposal,provider=meta,evidence=r['evidence']))
  if proposal['decision']=='patch' and 'registry' in state:
   plan=proposal['iteration_plan'];write(rd/('base_selection_'+str(r['model_attempts'])+'.json'),dict(plan=plan,comparators=evidence['iteration_context'],evidence=r['evidence']))
   selected=plan['base_id']
   if selected!=r['parent'].get('pool_id'):
    r.update(parent=copy.deepcopy(state['registry'][selected]),phase='capture',feedback={'type':'base_switched','message':'Re-author code against newly supplied source; no old patch was applied'})
    self.save(state);return True
   r['base_selection']=plan
  if proposal['decision']=='stop':
   self.finish(state,r,'no_supported_patch','model_stopped',analysis=proposal['failure_reason']);state['status']='model_stopped';self.save(state);return False
  if proposal['decision']=='diagnose':
   if r['diagnostic_turns']>=self.config['budgets']['max_diagnostics_per_round']:
    self.finish(state,r,'rejected','diagnostic_budget_exhausted');return True
   try:new,ids=diagnostics(evidence,proposal['diagnostics'],self.config['context']['max_chars'])
   except AgentError as e:
    r['feedback']=dict(type='diagnostic_request_rejected',message=str(e));r['diagnostic_turns']+=1;self.save(state);return True
   r['diagnostic_turns']+=1;path=rd/('evidence_'+str(r['diagnostic_turns']).zfill(2)+'.json');write(path,new);r['evidence']=asset(path);r['feedback']=dict(type='diagnostic_result',new_evidence_ids=ids);self.save(state);return True
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
  stages=self.adapter.stages(rd,r['bundle'],r['attempt'],self.config)
  if r['stage_index']>=len(stages):r['phase']='evaluate';self.save(state);return
  stage=stages[r['stage_index']];tag=stage['name']+'_a'+str(r['attempt_number'])+'_j'+str(r['job_retry']);folder=rd/'jobs'/tag
  inputs={**read(self.root/'host_freeze.json')['code'],**stage['inputs']}
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
    if not stage['gpu']:
     r['offline_retries']=r.get('offline_retries',0)+1
     if r['offline_retries']<=self.config['budgets']['max_infrastructure_attempts']:
      r['job_retry']+=1;self.save(state);return
     self.finish(state,r,'rejected','offline_task_interrupted_repeatedly',failure=r['last_job_failure']);return
    if r['attempt_number']<self.config['budgets']['max_infrastructure_attempts']:
     r['attempt_number']+=1;r['attempt']=self.adapter.attempt(rd,r['bundle'],r['attempt_number']);r['job_retry']=0
     # All candidate/causal evidence remains pinned. A new harness and output
     # preserve every partial episode; no neural-history splice or replay.
     new_stages=self.adapter.stages(rd,r['bundle'],r['attempt'],self.config)
     r['stage_index']=next(i for i,s in enumerate(new_stages) if s['name']=='harness_cpu');self.save(state);return
    state['status']='blocked_infrastructure';r['status']='infrastructure_unresolved';self.save(state);return
   self.finish(state,r,'rejected','offline_validation_failed',failure=r['last_job_failure']);return
  try:self.adapter.validate_stage(stage['name'],rd,r['bundle'],r['attempt'])
  except (AgentError,ValueError,KeyError,TypeError) as e:
   self.finish(state,r,'rejected','offline_proof_rejected',stage=stage['name'],error=str(e)[:2000]);return
  r['stage_index']+=1;r['job_retry']=0;r.pop('current_job',None);self.save(state);event(self.root,'stage_passed',round=r['id'],stage=stage['name'],receipt=asset(folder/'receipt.json'))
 def evaluate_step(self,state,r):
  rd=self.round_dir(r)
  try:
   metrics,strict=self.adapter.results(r['bundle'],r['attempt'])
   verdict=acceptance(metrics,state['champion']['metrics'],state['reference_metrics'],self.config['acceptance'],strict)
   if 'registry' in state:
    verdict=governance.retain_gains(verdict,metrics,r['parent'],state['registry'])
    governance.add_result(state['registry'],r['parent'],r['bundle'],r['attempt']['output'],metrics)
    write(self.root/'candidate_registry.json',state['registry'])
  except (AgentError,ValueError,KeyError,TypeError) as e:
   self.finish(state,r,'rejected','invalid_evaluation',error=str(e)[:2000]);return
  verdict.update(simulation_only=state['simulation_only'],GPU_executed=not state['simulation_only'])
  write(rd/'acceptance.json',dict(**verdict,metrics=metrics,strict_evaluation=asset(Path(r['attempt']['output'])/'evaluation.json')))
  r.update(verdict=verdict,metrics=metrics,phase='presentation');self.save(state)
  if not self.config.get('presentation',{}).get('enabled',False) or state['simulation_only']:self.finalize_evaluation(state,r)

 def finalize_evaluation(self,state,r):
  rd=self.round_dir(r);verdict=r['verdict'];metrics=r['metrics']
  if verdict['accepted']:
   old=copy.deepcopy(state['champion']);b=r['bundle']
   champion=dict(version=r['id'],code=b['code'],release=b['release'],release_name=b['release_name'],code_pins=b['candidate_pins'],matrix=r['attempt']['output'],metrics=metrics,reference_metrics=state['reference_metrics'],acceptance=asset(rd/'acceptance.json'))
   # Write the ledger and pointer before advancing to another model decision.
   write(rd/'promotion.json',dict(previous=old,candidate=champion,acceptance=asset(rd/'acceptance.json'),scope='local working champion only',production_driver_changed=False,global_patch_validated=False))
   if 'registry' in state:champion.update(pool_id=r['id'],profile=r['parent']['profile'])
   state['champion']=champion;write(self.root/'working_champion.json',champion)
  self.finish(state,r,'accepted' if verdict['accepted'] else 'tradeoff_retained' if verdict.get('gain_retention') and 'lost_high_score_branch_gain' in verdict['reasons'] else 'rejected','paired_acceptance_passed' if verdict['accepted'] else 'paired_acceptance_failed',acceptance=verdict,media=r.get('media'))

 def presentation_step(self,state,r):
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
   if (state.get('active_round') or {}).get('current_job'):jobs.cancel(Path(state['active_round']['current_job']))
   state['status']='stopped';self.save(state);return False
  verify_pins(state['champion']['code_pins'])
  r=state['active_round'] or self.new_round(state)
  if r.get('retry_after',0)>time.time():return True
  r.pop('retry_after',None)
  if r['phase']=='capture':
   latest=state['history'][-1].get('matrix') if state['history'] else None
   verify_pins(r['parent']['code_pins'])
   cfg=copy.deepcopy(self.config)
   if 'registry' in state:cfg['evidence_roots']+=list({v['matrix'] for v in state['registry'].values()})
   e=collect(cfg,state['champion'],state['history'],latest)
   if 'registry' in state:
    parent=r['parent'];e['selected_parent_profile']=parent['profile'];e['selected_parent_id']=parent['pool_id'];source=Path(parent['code'])/'joint_safety.py';e['items']['source'].update(asset=asset(source),text=source.read_text())
    e=governance.attach(e,state['registry'],state['champion'],state['reference_metrics'])
    from evidence import bounded
    e=bounded(e,self.config['context']['max_chars'])
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
