"""Icebreaking, search patience and qualification are separate durable dimensions."""
import copy,math
from core import AgentError,BudgetExceeded,digest,finite,remaining,read,verify
from final_verdict import healthy,validate

def enabled(config):return config.get('host_governance',{}).get('enabled',False)
def settings(config):
 p=config['host_governance']
 for k in ['search_patience','max_effective_evaluations_per_stage','max_infrastructure_retries_per_stage','max_validation_pools']:
  if type(p.get(k)) is not int or p[k]<=0:raise AgentError('Invalid host quota: '+k)
 return p

def ensure(state,config):
 p=settings(config);task=state['active_task_id'];ledger=state['task_convergence'][task]
 key=digest({'task':task,'experiment':ledger['experiment_fingerprint'],'regressions':sorted((x['task_id'],x['convergence']['experiment_fingerprint']) for x in state.get('task_archive',[]) if x['task_id']!=task)})
 stages=state.setdefault('search_stages',{})
 if key not in stages:
  # The carried policy has its historical admission; a failing new task does not.
  baseline=None if state.get('counterexample_training_task_id')==task or ledger.get('recovery_baseline') or not all(healthy(x) for x in state['champion']['metrics'].values()) else copy.deepcopy(state['champion'])
  if baseline is not None:
   target=ledger['policy']['target_case'];valid=[x for x in state.get('registry',{}).values() if __import__('governance').qualified_entry(x) and target in x['metrics'] and all(healthy(m) for m in x['metrics'].values())]
   baseline=copy.deepcopy(max([baseline,*valid],key=lambda x:x['metrics'][target]['score']['hdscore']))
  target=ledger['policy']['target_case']
  stages[key]={'task':task,'phase':'icebreaking' if baseline is None else 'convergence','best_observed_candidate':copy.deepcopy(state['champion']),'best_observed_score':state['champion']['metrics'][target]['score']['hdscore'],'best_qualified_candidate':baseline,'best_qualified_score':baseline['metrics'][target]['score']['hdscore'] if baseline else None,'evaluations':{},'effective_evaluations':0,'infra_events':{},'infrastructure_retries':0,'search_stagnation':0,'search_stopped':False,'stop_reason':None}
 state['active_search_stage']=key
 return stages[key]

def before_round(state,config):
 s=ensure(state,config);p=settings(config)
 if s['search_stopped']:raise BudgetExceeded('Search stopped: '+str(s['stop_reason']))
 if s['effective_evaluations']>=p['max_effective_evaluations_per_stage']:raise BudgetExceeded('Effective evaluation stage quota exhausted')
 return s

def observe(state,config,round_id,candidate,metrics,verdict):
 s=ensure(state,config);p=settings(config);receipt=digest({'candidate':candidate,'metrics':metrics,'verdict':verdict})
 if round_id in s['evaluations']:
  if s['evaluations'][round_id]['receipt']!=receipt:raise AgentError('Conflicting search evaluation retry')
  return s['evaluations'][round_id]
 # Only host-evaluated algorithms reach here; transport/CPU failures never call it.
 prior=s['best_qualified_score'];qualified=validate(verdict,metrics,candidate.get('code_pins')); target=state['task_convergence'][state['active_task_id']]['policy']['target_case'];score=metrics.get(target,{}).get('score',{}).get('hdscore')
 gain=score-prior if finite(score) and prior is not None else None
 s['effective_evaluations']+=1
 if finite(score) and score>s['best_observed_score']:s.update(best_observed_candidate=copy.deepcopy(candidate),best_observed_score=score)
 row={'receipt':receipt,'qualified':qualified,'score':score,'best_qualified_score_before':prior,'gain_vs_best_qualified':gain,'phase_before':s['phase']}
 if prior is None:
  # No patience debit during icebreaking, including eight rejected attempts.
  if qualified:
   s.update(best_qualified_candidate=copy.deepcopy(candidate),best_qualified_score=score,phase='convergence',search_stagnation=0);row['established_first_qualified']=True
 else:
  material=qualified and finite(gain) and gain>=state['task_convergence'][state['active_task_id']]['policy']['minimum_gain']-1e-9
  s['search_stagnation']=0 if material else s['search_stagnation']+1
  if qualified and finite(score) and score>prior:
   s.update(best_qualified_candidate=copy.deepcopy(candidate),best_qualified_score=score)
  if s['search_stagnation']>=p['search_patience']:s.update(search_stopped=True,stop_reason='search_patience_exhausted')
 if s['effective_evaluations']>=p['max_effective_evaluations_per_stage']:s.update(search_stopped=True,stop_reason='stage_evaluation_quota_exhausted')
 s['evaluations'][round_id]=row
 # Search exhaustion is not convergence or permission to validate/publish.
 return row

def infrastructure_failure(state,config,event_id):
 s=ensure(state,config)
 if event_id not in s['infra_events']:
  s['infra_events'][event_id]=True;s['infrastructure_retries']+=1
 if s['infrastructure_retries']>settings(config)['max_infrastructure_retries_per_stage']:
  s.update(search_stopped=True,stop_reason='infrastructure_retry_quota_exhausted');return False
 return True

def resource_plan(baseline_case_count,minimum_evaluations,validation_case_count,seconds_per_episode,safety_factor=1.25,training_episodes_per_dispatch=1):
 if any(type(v) is not int or v<0 for v in [baseline_case_count,minimum_evaluations,validation_case_count]) or baseline_case_count<1:raise AgentError('Invalid resource case counts')
 if type(training_episodes_per_dispatch) is not int or training_episodes_per_dispatch<1:raise AgentError('Invalid host dispatch batch bound')
 if not finite(seconds_per_episode) or seconds_per_episode<=0 or not finite(safety_factor) or safety_factor<1:raise AgentError('Invalid resource calibration')
 episodes=baseline_case_count*minimum_evaluations+validation_case_count
 return {'minimum_evaluations':minimum_evaluations,'baseline_case_count':baseline_case_count,'validation_case_count':validation_case_count,'episodes':episodes,'gpu_dispatches':math.ceil(baseline_case_count/training_episodes_per_dispatch)*minimum_evaluations+validation_case_count,'wall_seconds':math.ceil(episodes*seconds_per_episode*safety_factor),'estimate_not_guarantee':True}

def admission(state,config,host_resource_profile=None):
 s=ensure(state,config);p=settings(config);ledger=state['task_convergence'][state['active_task_id']];m=p['resource_model']
 needed=(m['minimum_icebreaking_evaluations']+p['search_patience']) if s['best_qualified_candidate'] is None else max(1,p['search_patience']-ledger['small_gain_streak'])
 cases=len(state['champion']['metrics'])+sum(len(a['champion']['metrics']) for a in state.get('task_archive',[]) if a['task_id']!=state['active_task_id'])
 elapsed=m['seconds_per_episode']
 for row in [*state['champion']['metrics'].values(),*[x for a in state.get('task_archive',[]) for x in a['champion']['metrics'].values()]]:
  if row.get('summary'):
   verify(row['summary']);seconds=read(row['summary']['path']).get('wall_seconds')
   if finite(seconds) and seconds>0:elapsed=max(elapsed,seconds)
 batch=(host_resource_profile or {}).get('training_episodes_per_dispatch',1)
 plan=resource_plan(cases,needed,max(m['validation_case_estimate'],sum(p.get('stratification',{}).get('topology_quotas',{}).values())),elapsed,m['safety_factor'],batch)
 # Early check once per stage; future debits still use all original hard caps.
 available={'effective_evaluations':p['max_effective_evaluations_per_stage']-s['effective_evaluations'],'rounds':config['budgets']['max_rounds']-state['rounds_started'],'api_requests':config['budgets']['max_api_requests']-state['usage']['api_requests'],'gpu_dispatches':config['budgets']['max_gpu_attempts']-state['usage'].get('gpu_attempts',0),'wall_seconds':remaining(state,config),'tokens':config['budgets']['max_api_tokens']-state['usage']['charged_tokens']}
 passed=available['effective_evaluations']>=needed and available['rounds']>=needed and available['api_requests']>=needed and available['gpu_dispatches']>=plan['gpu_dispatches'] and available['wall_seconds']>=plan['wall_seconds'] and available['tokens']>=needed*m.get('minimum_tokens_per_evaluation',1)
 return {'passed':passed,'minimum_resource_plan':plan,'available':available,'no_budget_extension':True}


def admit_validation_pool(state,config,pool_fingerprint,case_count):
 p=settings(config);seen=state.setdefault('validation_pool_attempts',{})
 if pool_fingerprint in seen:return {'passed':True,'already_reserved':True}
 if state.get('legacy_validation_pool_count',0)+len(seen)>=p['max_validation_pools']:return {'passed':False,'reason':'validation_pool_quota_exhausted'}
 model=p['resource_model'];plan=resource_plan(1,0,case_count,model['seconds_per_episode'],model['safety_factor'])
 if config['budgets']['max_gpu_attempts']-state['usage'].get('gpu_attempts',0)<case_count or remaining(state,config)<plan['wall_seconds']:return {'passed':False,'reason':'insufficient_validation_resources'}
 # This is durable quota reservation only, not evidence of validation success.
 seen[pool_fingerprint]={'reserved':True};return {'passed':True,'already_reserved':False}
