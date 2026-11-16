"""Host-only predeclared strata. Never rebalance using candidate outcomes."""
import random
from collections import Counter
from core import AgentError,digest

def validate(pool,policy):
 if not policy.get('required'):return True
 cases=pool.get('cases',[]);counts=Counter(c.get('interaction_topology') for c in cases)
 if len({c.get('scene') for c in cases})<policy['minimum_distinct_scenes']:raise AgentError('Insufficient independent strata coverage')
 if any(counts[k]!=n for k,n in policy['topology_quotas'].items()) or set(counts)!=set(policy['topology_quotas']):raise AgentError('Frozen topology quota mismatch')
 severity=Counter(c.get('severity_band') for c in cases)
 if policy.get('severity_quotas') and dict(severity)!=policy['severity_quotas']:raise AgentError('Frozen severity quota mismatch')
 if not pool.get('sampling_seed') or pool.get('selection_uses_candidate_outcomes') is not False:raise AgentError('Pool sampling was not independently frozen')
 return True

def sample(candidates,excluded_scenes,policy,seed):
 rng=random.Random(seed);available=[c for c in candidates if c['scene'] not in set(excluded_scenes)]
 if any(set(c)&{'score','scores','passed','candidate_outcome','validation_result'} for c in available):raise AgentError('Outcome-dependent sampling metadata forbidden')
 rng.shuffle(available);slots=[t for t,n in sorted(policy['topology_quotas'].items()) for _ in range(n)]
 def choose(index,chosen,used):
  if index==len(slots):
   if policy.get('severity_quotas') and dict(Counter(c.get('severity_band') for c in chosen))!=policy['severity_quotas']:return None
   return chosen
  for c in available:
   if c['interaction_topology']!=slots[index] or c['scene'] in used:continue
   if policy.get('severity_quotas') and sum(x.get('severity_band')==c.get('severity_band') for x in chosen)>=policy['severity_quotas'].get(c.get('severity_band'),0):continue
   result=choose(index+1,chosen+[c],used|{c['scene']})
   if result is not None:return result
  return None
 chosen=choose(0,[],set())
 if chosen is None:raise AgentError('Awaiting independently sampled stratum assets')
 pool={'cases':chosen,'sampling_seed':str(seed),'selection_uses_candidate_outcomes':False};validate(pool,policy);pool['sampling_commitment']=digest(pool);return pool
