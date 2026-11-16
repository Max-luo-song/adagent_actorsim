"""Persistent iteration policy: retain gains, repair minimally, compare all contenders."""

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

import copy,difflib
pass  # Path is provided by the portable boundary above
from core import AgentError,asset,code_pins,finite,read,sha,verify,verify_pins
from validation import actual_metrics
PROFILES={
 'v17_spatial':{'methods':{'JointReference.retime_selected','JointReference.retiming_cost','JointReference.execution_benefit'},'extra_round':'v17_runtime_r2','extra_test':'test_execution_horizon.py','old_code':'light-execution-horizon-v17-r2'},
 'v15_joint':{'methods':{'JointReference.rollout'},'extra_round':'v13_to_v15_tracking_curvature_r2','extra_test':'test_tracking_curvature.py','old_code':'light-tracking-curvature-v15-r2'},
}
PROFILES['v15_corridor']={**PROFILES['v15_joint'],'methods':{'JointReference.rollout','JointReference.guides','JointReference.speed_targets'}}
SHARED=('scene','edited','seed','controls','initial_context','scene_manifest','map','scenario','target_track','vehicle','worker','native_collision','tracker','evaluator','termination','source_assets')
def healthy(metrics):
 for x in metrics.values():
  if x['status']!='completed' or x['controls']!=188 or x['observations']!=189 or x['collision'] or x['map_departure'] or x['unresolved_stop']:return False
  if not finite(x['score']['hdscore']):return False
 return True

def initial_pool(config,champion):
 pool={}
 for entry in config['candidate_registry']:
  matrix=Path(entry['matrix']);plan=read(matrix/'matrix.json');verify_pins(plan['code_sha256']);verify(plan['source_release']);verify(plan['source_patch_cpu'])
  tested=read(plan['source_patch_cpu']['path']);cp=code_pins(entry['code'])
  if not tested['passed'] or tested['code_sha256']!=cp:raise AgentError('Registry code differs from tested release: '+entry['id'])
  metrics=actual_metrics(matrix)
  if not healthy(metrics):raise AgentError('Registry seed has unhealthy physical execution: '+entry['id'])
  for case in ('original551','cutin551'):
   a=read(metrics[case]['contract']['path']);b=read(champion['metrics'][case]['contract']['path'])
   if any(a[k]!=b[k] for k in SHARED):raise AgentError('Registry scenarios/exogenous inputs differ: '+entry['id'])
  p=dict(version=entry['id'],pool_id=entry['id'],code=entry['code'],release=entry['release'],release_name=Path(entry['release']).name,matrix=entry['matrix'],code_pins=cp,metrics=metrics,profile=entry['profile'],eligible=entry['eligible'],evaluated_repairs=0,reference_metrics=champion['reference_metrics'])
  if p['eligible'] and p['profile'] not in PROFILES:raise AgentError('Unregistered base profile')
  pool[p['pool_id']]=p
 return pool

def dominates(a,b,tol=1e-9):
 x=a['metrics']['cutin551'];y=b['metrics']['cutin551']
 axes=[(x['score']['hdscore'],y['score']['hdscore']),(-x['final_abs_lateral_m'],-y['final_abs_lateral_m']),(-x['peak_abs_lateral_m'],-y['peak_abs_lateral_m'])]
 return all(u>=v-tol for u,v in axes) and any(u>v+tol for u,v in axes)
def frontier(pool):
 ids=[key for key,p in pool.items() if p.get('eligible') and healthy(p['metrics'])]
 return sorted(key for key in ids if not any(dominates(pool[other],pool[key]) for other in ids if other!=key))
def best_score(pool):return max(p['metrics']['cutin551']['score']['hdscore'] for p in pool.values() if p.get('eligible') and healthy(p['metrics']))

def version_aliases(pool,champion):
 matches={key:{key} for key in pool}
 def add(alias,key):
  if alias in pool and alias!=key:return
  matches.setdefault(alias,set()).add(key)
 for key,item in pool.items():
  for alias in (item.get('version'),item.get('release_name'),Path(item['code']).name if item.get('code') else None):
   if alias:add(alias,key)
  if item.get('code') and item.get('code')==champion.get('code') and item.get('code_pins')==champion.get('code_pins'):add(champion['version'],key)
 # Shared labels are not unique identities. They remain unknown, never guessed.
 return {alias:next(iter(keys)) for alias,keys in matches.items() if len(keys)==1}

def context(pool,champion,reference):
 ids=frontier(pool);tol=1e-9;bound=.25
 repair=[]
 for key in ids:
  p=pool[key];x=p['metrics']['cutin551'];w=champion['metrics']['cutin551'];r=reference['cutin551']
  side_effect=any(x[k]>min(w[k],r[k])+bound for k in ('final_abs_lateral_m','peak_abs_lateral_m'))
  if x['score']['hdscore']>w['score']['hdscore']+tol and side_effect and p.get('evaluated_repairs',0)==0:repair.append(key)
 required=max(repair,key=lambda k:pool[k]['metrics']['cutin551']['score']['hdscore']) if repair else None
 return dict(version_aliases=version_aliases(pool,champion),allowed_comparison_ids=sorted(pool),frontier_ids=ids,required_base=required,protected_target_hdscore=best_score(pool),working_version=champion.get('pool_id',champion['version']),rules=['Repair a higher-scoring branch before abandoning it','Lateral side effect does not mean physical safety failure','Retain Pareto alternatives; do not silently change objective or baseline','Compare against selected parent, working champion and high-score frontier','Tradeoff candidates are retained and labeled, not called full upgrades'],profile_methods={key:sorted(PROFILES[pool[key]['profile']]['methods']) for key in ids},discriminating_evidence_ids=['registry_'+key for key,p in pool.items() if p.get('evaluated_round')])

def select_parent(pool,champion,reference):
 ctx=context(pool,champion,reference)
 if ctx['required_base']:return copy.deepcopy(pool[ctx['required_base']])
 key=champion.get('pool_id')
 if key in ctx['frontier_ids']:return copy.deepcopy(pool[key])
 feasible=[k for k in ctx['frontier_ids'] if all(pool[k]['metrics']['cutin551'][m]<=min(champion['metrics']['cutin551'][m],reference['cutin551'][m])+.25 for m in ('final_abs_lateral_m','peak_abs_lateral_m'))]
 selected=max(feasible or ctx['frontier_ids'],key=lambda k:pool[k]['metrics']['cutin551']['score']['hdscore'])
 return copy.deepcopy(pool[selected])

def validate_plan(plan,evidence):
 ctx=evidence['iteration_context'];pool=evidence['base_pool']
 required={'base_id','strategy','why_this_base','preserve','fix_only','compared_versions','protected_score','rollback_evidence_ids'}
 if not isinstance(plan,dict) or set(plan)!=required:raise AgentError('Explicit iteration plan required')
 base=plan['base_id']
 if base not in ctx['frontier_ids']:raise AgentError('Base is not a healthy Pareto contender')
 if ctx['required_base'] and base!=ctx['required_base']:raise AgentError('Minimal repair of high-score branch required before changing base: '+ctx['required_base'])
 if not finite(plan['protected_score']) or abs(plan['protected_score']-ctx['protected_target_hdscore'])>1e-9:raise AgentError('Cannot forget or lower the best measured score')
 if plan['strategy'] not in ('minimal_repair','alternative_branch'):raise AgentError('Unregistered iteration strategy')
 if ctx['required_base'] and plan['strategy']!='minimal_repair':raise AgentError('First repair must be minimal')
 for key in ('why_this_base','preserve','fix_only'):
  if not isinstance(plan[key],str) or not plan[key].strip():raise AgentError('Iteration rationale/gain preservation must be explicit')
 compared=plan['compared_versions']
 if not isinstance(compared,list) or any(not isinstance(x,str) for x in compared):raise AgentError('compared_versions must be an array of canonical version IDs')
 missing=sorted(set(ctx['frontier_ids'])-set(compared));unknown=sorted(set(compared)-set(pool))
 if missing or unknown:raise AgentError('Version comparison mismatch: missing_required='+str(missing)+'; unknown_ids='+str(unknown)+'; allowed_ids='+str(sorted(pool))+'; aliases='+str(ctx.get('version_aliases',{})))
 if not isinstance(plan['rollback_evidence_ids'],list) or any(k not in evidence['items'] for k in plan['rollback_evidence_ids']):raise AgentError('Unknown rollback evidence')
 if plan['strategy']=='alternative_branch' and not set(plan['rollback_evidence_ids']).intersection(ctx['discriminating_evidence_ids']):raise AgentError('Changing route requires a completed minimal-repair experiment, not a narrative')
 return plan

def attach(evidence,pool,champion,reference):
 e=copy.deepcopy(evidence);e['iteration_context']=context(pool,champion,reference);e['base_pool']={}
 for key,p in pool.items():
  e['base_pool'][key]={k:p[k] for k in ('version','profile','eligible','evaluated_repairs','metrics')}
  if p.get('evaluated_round'):e['base_pool'][key].update(evaluated_round=True,parent_id=p['parent_id'])
  e['items']['registry_'+key]={'facts':e['base_pool'][key],'source_code':asset(Path(p['code'])/'joint_safety.py')}
 profile=e.get('selected_parent_profile')
 if profile in PROFILES:e['items']['source']['editable_methods']=sorted(PROFILES[profile]['methods'])
 return e

def minimality(before,after,changed,policy):
 lines=sum(line.startswith(('+','-')) and not line.startswith(('+++','---')) for line in difflib.unified_diff(before.splitlines(),after.splitlines()))
 if len(changed)>policy['maximum_changed_methods'] or lines>policy['maximum_changed_lines']:raise AgentError('Patch changes too many mechanisms; split into minimal experiments')
 return dict(changed_methods=changed,changed_lines=lines)

def retain_gains(verdict,metrics,parent,pool):
 result=copy.deepcopy(verdict);target=metrics['cutin551']['score']['hdscore'];best=best_score(pool);parent_score=parent['metrics']['cutin551']['score']['hdscore']
 result['gain_retention']=dict(selected_parent=parent.get('pool_id',parent['version']),parent_hdscore=parent_score,previous_best_hdscore=best,candidate_hdscore=target,gain_vs_parent=target-parent_score,gain_vs_best=target-best)
 if target<best-1e-9:
  result['accepted']=False;result['reasons'].append('lost_high_score_branch_gain')
 result['classification']='full_paired_improvement' if result['accepted'] else 'tradeoff_or_rejected'
 return result

def add_result(pool,parent,bundle,matrix,metrics):
 if parent.get('pool_id') in pool:pool[parent['pool_id']]['evaluated_repairs']+=1
 rid=bundle['round_id'];pool[rid]=dict(version=rid,pool_id=rid,code=bundle['code'],release=bundle['release'],release_name=bundle['release_name'],matrix=matrix,code_pins=bundle['candidate_pins'],metrics=copy.deepcopy(metrics),profile=parent['profile'],eligible=bool(healthy(metrics) and metrics['original551']['score']['hdscore']>=parent['reference_metrics']['original551']['score']['hdscore']-1e-9 and metrics['original551']['score']['pdms']>=parent['reference_metrics']['original551']['score']['pdms']-1e-9 and metrics['original551']['progress_m']>=parent['reference_metrics']['original551']['progress_m']*.95 and metrics['original551']['stopped_s']<=parent['reference_metrics']['original551']['stopped_s']+1.),evaluated_repairs=0,reference_metrics=parent['reference_metrics'],evaluated_round=True,parent_id=parent['pool_id'])
