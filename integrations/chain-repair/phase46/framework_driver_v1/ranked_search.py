"""Reorder expansion only using current exact seed costs and feasibility."""
import math
def candidates(search,seeds,maxiter,evaluate):
 names=['warm','coast'] if any(n=='warm' for n,_ in seeds) else ['predicted_road_route','coast']
 primary=[x for n in names for x in seeds if x[0]==n]
 lateral=[x for n in ('escape_left','escape_right') for x in seeds if x[0]==n and n not in {k for k,_ in primary}];primary+=lateral
 search.audit.update(lateral_coverage_seeds=[n for n,_ in lateral],primary_seeds=[n for n,_ in primary],expansion_order_policy='current_exact_seed_feasibility_then_objective',candidate_pruning=False)
 for n,s in primary:yield n,s,maxiter,'consensus_primary'
 if search.agreement():return
 search.audit['expanded']=True;seen={n for n,_ in primary};ranked=[];audit=[]
 for index,(name,seed) in enumerate(seeds):
  if name in seen:continue
  gap,cost=evaluate(name,seed)
  valid=math.isfinite(gap) and math.isfinite(cost)
  ranked.append(((not valid,gap < -1e-5,cost if valid else math.inf,index),name,seed));audit.append(dict(seed=name,minimum_constraint=gap if math.isfinite(gap) else None,initial_objective=cost if math.isfinite(cost) else None,finite_ranking_input=valid))
 feasible_costs=[r['objective'] for r in search.rows if r.get('feasible') and math.isfinite(r.get('objective',math.inf))]
 if feasible_costs:
  best=min(feasible_costs);reserve=best+max(abs(best),1.)
  # Promote only already-feasible cost-competitive seeds. Leave all other
  # starts in their original order; seed scores are not lower bounds.
  promoted=[row for row in ranked if not row[0][0] and not row[0][1] and row[0][2]<=reserve]
  promoted.sort(key=lambda row:row[0]);promoted_names={row[1] for row in promoted};others=[row for row in ranked if row[1] not in promoted_names];ranked=promoted+others
  search.audit.update(seed_cost_reserve=reserve,unpromoted_order_preserved=True,promoted_seeds=[row[1] for row in promoted])
 else:ranked.sort(key=lambda row:row[0])
 search.audit.update(expansion_seed_evaluations=audit,expansion_order=[row[1] for row in ranked])
 for _,name,seed in ranked:
  yield name,seed,maxiter,'consensus_expansion'
  if search.agreement():return
