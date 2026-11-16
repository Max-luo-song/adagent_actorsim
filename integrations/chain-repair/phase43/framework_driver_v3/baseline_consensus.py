"""Full-budget converged-start agreement; no probe/restart duplication.

This is a bounded search policy, not a global optimum certificate. Every admitted
candidate passes the unchanged current constraints and stopping witness. If two
distinct primary starts do not converge to similarly good moving plans, search
all remaining original starts once at their original iteration budget.
"""
class ConsensusSearch:
 def __init__(self,policy,minimum_final_speed,neural_stop=False):
  self.policy=policy;self.minimum_final_speed=minimum_final_speed;self.neural_stop=neural_stop;self.rows=[];self.audit=dict(strategy='converged_start_agreement',expanded=False,global_optimum_claimed=False,iteration_budgets_reduced=False,restarts_discarded=False)
 def observe(self,row):self.rows.append(row)
 def acceptable(self,row):
  m=row.get('native_alignment_proxy',{});moving=row['states'][-1][3]>1e-6 and row['states'][-1][3]>=self.minimum_final_speed-1e-6
  return bool(row['feasible'] and row.get('solver_success') and not row.get('used_verified_seed') and (moving or self.neural_stop) and m.get('local_progress_proxy',0.)>=self.policy['minimum_progress_proxy'] and m.get('dac_proxy',0.)>=self.policy['minimum_dac_proxy'] and m.get('comfort_native_kinematic_flag',0.)==1.)
 def candidates(self,seeds,maxiter):
  names=['warm','coast'] if any(n=='warm' for n,_ in seeds) else ['predicted_road_route','coast'];primary=[x for n in names for x in seeds if x[0]==n]
  self.audit['primary_seeds']=[n for n,_ in primary]
  for n,s in primary:yield n,s,maxiter,'consensus_primary'
  good=[r for r in self.rows if self.acceptable(r)];distinct={r.get('start_key') for r in good};costs=[r['objective'] for r in good];spread=(max(costs)-min(costs))/max(abs(min(costs)),1.) if len(costs)>=2 else None
  passed=len(good)>=2 and len(distinct)>=2 and spread<=self.policy['maximum_refinement_improvement_fraction'];self.audit.update(agreement_passed=passed,relative_objective_spread=spread,qualified_primary_count=len(good))
  if passed:return
  self.audit['expanded']=True
  seen={n for n,_ in primary}
  for n,s in seeds:
   if n not in seen:yield n,s,maxiter,'consensus_expansion'
