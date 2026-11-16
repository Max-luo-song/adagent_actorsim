"""One host-owned qualification result. Ineligible observations cannot set anchors."""
import copy
from core import finite,digest,AgentError
KEYS=('hdscore','pdms','rc')
def healthy(row):
 return (row.get('status')=='completed' and type(row.get('controls')) is int and row['controls']>0 and row.get('observations')==row['controls']+1 and all(row.get(k) is False for k in ('collision','map_departure','unresolved_stop')) and all(finite(row.get('score',{}).get(k)) for k in KEYS))

def finalize(raw,metrics,baseline,tolerance,retention_passed,motion,candidate_pins=None):
 if not finite(tolerance) or not 0<=tolerance<=1e-9:raise AgentError('Qualification permits numerical tolerance only')
 reasons=list(raw.get('reasons',[]));adapter_blockers=[r for r in reasons if not r.endswith(':no_incremental_target_gain')]
 violations=[]
 if set(metrics)!=set(baseline):violations.append('case_set_changed')
 for name,n in metrics.items():
  if not healthy(n):violations.append(name+':incomplete_unsafe_or_unscored');continue
  b=baseline.get(name,{})
  if healthy(b):
   if n['controls']!=b['controls']:violations.append(name+':unequal_horizon')
   for k in KEYS:
    if n['score'][k]<b['score'][k]-tolerance:violations.append(name+':'+k+'_regression')
 gates={'adapter':{'passed':not adapter_blockers,'reasons':adapter_blockers},'per_case_nonregression':{'passed':not violations,'reasons':violations},'historical_retention':{'passed':retention_passed is True},'motion':copy.deepcopy(motion)}
 qualified=all(x.get('passed') is True for x in gates.values())
 reasons+=violations
 if retention_passed is not True:reasons.append('previous_task_nonregression_unverified')
 if motion.get('passed') is not True:reasons.append('host_motion_gate_failed')
 binding={'metrics_sha256':digest(metrics),'baseline_sha256':digest(baseline),'gates':gates,'candidate_pins':copy.deepcopy(candidate_pins or {})}
 result=dict(raw,candidate_pins=binding['candidate_pins'],metrics_sha256=binding['metrics_sha256'],baseline_sha256=binding['baseline_sha256'],schema='final_candidate_verdict_v1',qualified=qualified,accepted=bool(raw.get('accepted') and qualified),reasons=list(dict.fromkeys(reasons)),gates=gates,qualification_basis=digest(binding))
 return result

def validate(verdict,metrics=None,candidate_pins=None):
 if not isinstance(verdict,dict) or verdict.get('schema')!='final_candidate_verdict_v1' or verdict.get('qualified') is not True:return False
 if not all(verdict.get('gates',{}).get(k,{}).get('passed') is True for k in ['adapter','per_case_nonregression','historical_retention','motion']):return False
 if metrics is not None and verdict.get('metrics_sha256')!=digest(metrics):return False
 if candidate_pins is not None and verdict.get('candidate_pins')!=candidate_pins:return False
 binding={k:verdict.get(k) for k in ['metrics_sha256','baseline_sha256','gates','candidate_pins']}
 return verdict.get('qualification_basis')==digest(binding)
