"""Real v9 initialization needs host-issued lineage; upgrading never resets usage."""
import copy
from core import AgentError,read,verify

def inherit(state,config):
 binding=config.get('budget_lineage')
 if not binding:raise AgentError('Real v9 initialization requires a host-issued budget lineage receipt; no automatic new budget')
 verify(binding);proof=read(binding['path'])
 if proof.get('schema')!='host_budget_lineage_v1':raise AgentError('Unknown budget lineage receipt')
 verify(proof['previous_state']);verify(proof['previous_config']);previous=read(proof['previous_state']['path']);old=read(proof['previous_config']['path'])
 if config['budgets']!=old['budgets']:
  allocation=proof.get('explicit_operator_allocation')
  if not allocation:raise AgentError('Budget changes require a separate explicit operator allocation')
  verify(allocation)
  if read(allocation['path']).get('approved_cumulative_budgets')!=config['budgets']:raise AgentError('Operator allocation differs from cumulative budgets')
 known=previous.get('validation_pool_attempts')
 prior=proof.get('prior_validation_pool_count',len(known) if known is not None else None)
 if type(prior) is not int or prior<len(known or {}):raise AgentError('Legacy migration needs host-attested prior pool usage; it cannot reset pool quota')
 state['legacy_validation_pool_count']=prior-len(known or {})
 state['usage']=copy.deepcopy(previous['usage']);state['rounds_started']=previous['rounds_started'];state['started_epoch']=previous['started_epoch']
 state['validation_pool_attempts']=copy.deepcopy(previous.get('validation_pool_attempts',{}));state['budget_lineage']={'receipt':binding,'counters_reset':False,'previous_counters_preserved':True}
