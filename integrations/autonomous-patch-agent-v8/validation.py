"""Acceptance is deterministic and inaccessible to model edits."""

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

import copy
pass  # Path is provided by the portable boundary above
from core import AgentError,asset,finite,read,verify,verify_pins
CASES=('original551','cutin551')

def actual_metrics(matrix):
 matrix=Path(matrix);plan=read(matrix/'matrix.json');metrics={}
 for case,arm in [('original551','A'),('cutin551','B')]:
  work=Path(plan['cases'][case]['arms']['on']);receipt=read(work/'completion.json')
  if receipt['source_pins']!=plan['code_sha256']:raise AgentError('Completion source pins differ')
  for k in ('report','summary','contract'):verify(receipt[k])
  if read(receipt['report']['path']).get('experiment_plan')!=asset(work/'plan.json'):raise AgentError('Report experiment plan mismatch')
  r=read(receipt['report']['path']);summary=read(receipt['summary']['path']);contract=read(receipt['contract']['path'])
  if r.get('validation_contract')!=asset(work/'contract.json') or r.get('validation_patch')!='on':raise AgentError('Execution contract is not bound to report')
  if not r.get('real_closed_loop') or not r.get('model_weights_unchanged'):raise AgentError('Not a genuine frozen-model closed loop')
  if not contract.get('fresh_history') or contract.get('score_kind')!='executed_path':raise AgentError('Need fresh-history executed-path metrics')
  values=summary.get('native_scores',{}).get('scores',{}).get('executed_path')
  if not values:raise AgentError('Native executed-path scores unavailable')
  score={k:values[k]['value'] for k in ('hdscore','pdms','rc')}
  positions=[abs(float(o['motion_state'][0])) for o in r.get('observations',[])]
  if not positions:raise AgentError('Actual final observation missing')
  item=dict(score=score,status=r['status'],controls=len(r.get('trace',[])),observations=len(r.get('observations',[])),collision=any(t.get('swept_collision',False) for t in r['trace']) or summary['anomalies']['environment_collision'],map_departure=any(o['environment'].get('authorized_static_map_departure',False) for o in r['observations']),unresolved_stop=summary['stop_recovery_unresolved'],progress_m=summary['progress_along_initial_forward_m'],stopped_s=summary['stopped_seconds'],final_abs_lateral_m=positions[-1],peak_abs_lateral_m=max(positions),report=receipt['report'],summary=receipt['summary'],contract=receipt['contract'])
  metrics[case]=item
 return metrics

def acceptance(candidate,champion,reference,policy,strict):
 reasons=[]
 for case in CASES:
  n=candidate[case];b=champion[case];ref=reference[case]
  for name,item in [('candidate',n),('champion',b),('reference',ref)]:
   vals=[*item['score'].values(),*[item[k] for k in ('progress_m','stopped_s','final_abs_lateral_m','peak_abs_lateral_m')]]
   if any(not finite(v) for v in vals):reasons.append(case+':'+name+'_invalid_metrics')
   if item['status']!='completed' or item['controls']!=188 or item['observations']!=189:reasons.append(case+':'+name+'_incomplete')
   if item['collision'] or item['map_departure'] or item['unresolved_stop']:reasons.append(case+':'+name+'_safety_or_stop_failure')
  if reasons and any(case+':' in s and ('invalid' in s or 'incomplete' in s) for s in reasons):continue
  if n['score']['pdms']<b['score']['pdms']-policy['numeric_tolerance']:reasons.append(case+':pdms_regression')
  if n['progress_m']<b['progress_m']*policy['min_progress_ratio'] or n['stopped_s']>b['stopped_s']+policy['max_extra_stopped_seconds']:reasons.append(case+':progress_or_stop_regression')
  if not policy.get('lateral_diagnostics_only',False):
   for key in ('final_abs_lateral_m','peak_abs_lateral_m'):
    if n[key]>min(b[key],ref[key])+policy['lateral_tolerance_m']:reasons.append(case+':'+key+'_regression')
  gain=n['score']['hdscore']-b['score']['hdscore']
  if case=='original551' and gain< -policy['original_max_hd_drop']-policy['numeric_tolerance']:reasons.append(case+':original_hd_regression')
  if case=='cutin551' and gain<=policy['numeric_tolerance']:reasons.append(case+':no_incremental_target_gain')
 return dict(accepted=not reasons,reasons=reasons,target_hd_gain=candidate['cutin551']['score']['hdscore']-champion['cutin551']['score']['hdscore'],target_hd_gain_vs_v13=candidate['cutin551']['score']['hdscore']-reference['cutin551']['score']['hdscore'],strict_matrix_status=strict.get('status'),strict_missing_cases=strict.get('missing_cases',[]),global_patch_validated=False,formal_certification=False,scope='working champion on admitted 551 original/cut-in pair only; no global deployment',production_driver_changed=False)
