"""Empirical plateau tracking for comparable, completed, nonregressing pairs only."""
import copy
from core import AgentError,digest,finite,read,verify
from governance import SHARED

def validate_config(policy):
 if not isinstance(policy.get('enabled'),bool):raise AgentError('convergence.enabled must be boolean')
 if not isinstance(policy.get('window'),int) or isinstance(policy['window'],bool) or policy['window']<2:raise AgentError('Convergence needs at least two valid evaluations')
 if not finite(policy.get('minimum_gain')) or not 0<policy['minimum_gain']<=1:raise AgentError('Invalid absolute HDScore convergence threshold')
 if not finite(policy.get('numeric_tolerance')) or not 0<=policy['numeric_tolerance']<policy['minimum_gain']:raise AgentError('Invalid convergence tolerance')
 if not isinstance(policy.get('target_case'),str):raise AgentError('Convergence target case required')

def fingerprint(task_id,metrics,simulation_only):
 inputs={}
 for case,m in metrics.items():
  if m.get('contract'):
   verify(m['contract']);c=read(m['contract']['path']);inputs[case]={k:c[k] for k in SHARED}
   if not c.get('fresh_history') or c.get('score_kind')!='executed_path':raise AgentError('Convergence requires fresh executed-path evaluation')
  elif simulation_only:inputs[case]={'test_fixture_only':True}
  else:raise AgentError('Convergence cannot use unbound metrics')
 return digest({'task':task_id,'inputs':inputs})

def initial(task_id,baseline,policy,simulation_only=False):
 validate_config(policy)
 if len(baseline)<2 or any(m.get('status')!='completed' or m.get('controls',0)<1 or m.get('observations')!=m.get('controls',0)+1 or any(m.get(k) for k in ('collision','map_departure','unresolved_stop')) for m in baseline.values()):raise AgentError('Convergence needs a complete healthy paired baseline')
 for m in baseline.values():
  values=[*m['score'].values(),*[m[k] for k in ('progress_m','stopped_s','final_abs_lateral_m','peak_abs_lateral_m')]]
  if any(not finite(v) for v in values):raise AgentError('Nonfinite paired convergence baseline')
 target=policy['target_case'];score=baseline[target]['score']['hdscore']
 if not finite(score):raise AgentError('Nonfinite convergence baseline')
 return dict(task_id=task_id,experiment_fingerprint=fingerprint(task_id,baseline,simulation_only),initial_score=score,best_score=score,window=[],observations=[],small_gain_streak=0,converged=False,policy=copy.deepcopy(policy))

def observe(ledger,round_id,metrics,verdict,best_before,simulation_only=False):
 p=ledger['policy'];target=p['target_case'];score=metrics[target]['score']['hdscore'];key=fingerprint(ledger['task_id'],metrics,simulation_only)
 receipt=digest(dict(round=round_id,metrics=metrics,verdict=verdict,best_before=best_before))
 previous=next((r for r in ledger['observations'] if r['round_id']==round_id),None)
 if previous:
  if previous['receipt']!=receipt:raise AgentError('Conflicting duplicate convergence observation')
  return previous
 if key!=ledger['experiment_fingerprint']:raise AgentError('Different scenario/edit/seed inputs cannot share a convergence window')
 reasons=verdict.get('reasons',[]);allowed={target+':no_incremental_target_gain'}
 complete=len(metrics)>=2 and all(m.get('status')=='completed' and m.get('controls',0)>0 and m.get('observations')==m.get('controls',0)+1 and not any(m.get(k) for k in ('collision','map_departure','unresolved_stop')) for m in metrics.values())
 eligible=complete and finite(score) and finite(best_before) and not(set(reasons)-allowed) and score>=best_before-p['numeric_tolerance']
 gain=max(0.,score-best_before) if verdict.get('accepted') else 0.
 row=dict(round_id=round_id,receipt=receipt,eligible=eligible,candidate_score=score,best_score_before=best_before,effective_gain=gain,accepted=verdict.get('accepted',False),reasons=reasons,simulation_only=simulation_only)
 ledger['observations'].append(row)
 if not eligible:
  ledger['window']=[];ledger['small_gain_streak']=0;row['classification']='completed_but_regressing_or_invalid';return row
 if verdict.get('accepted'):ledger['best_score']=max(ledger['best_score'],score)
 small=gain<p['minimum_gain']-p['numeric_tolerance'];row['classification']='small_gain' if small else 'material_gain'
 ledger['window']=(ledger['window']+[row])[-p['window']:] if small else []
 ledger['small_gain_streak']=len(ledger['window']);ledger['converged']=len(ledger['window'])>=p['window']
 return row
