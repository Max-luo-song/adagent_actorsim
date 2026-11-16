"""Model-facing allowlist; private validation artifacts are never evidence."""

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

import copy,re
pass  # Path is provided by the portable boundary above
DENY={'generalization','private_validation','validation_results','held_out_results','heldout_results','private_registry','sealed_cases','development_cases','missing_cases','strict_missing_cases','private_error','failure_scene','validation_reason','validation_score','private_attestation'}

def public_evidence(evidence,state,config,root):
 e=copy.deepcopy(evidence);cases=set(e.get('task_contract',{}).get('case_ids',['original551','cutin551']));scene=state.get('current_task_descriptor',{}).get('scene',config['generalization'].get('training_scene'))
 private=[str((Path(root)/'private_validation').resolve()),str(Path((config['generalization'].get('pool_catalog') or config['generalization'].get('private_registry'))['path']).resolve().parent)]
 def clean(value):
  if isinstance(value,dict):
   out={}
   for k,v in value.items():
    if k.lower() in DENY or k.lower().startswith(('private_validation_','sealed_validation_','heldout_')):continue
    if re.fullmatch(r'(?:original|cutin|scene)[0-9]+',k) and k not in cases and k!='scene'+str(scene):continue
    if k in ('metrics','legacy_metrics') and isinstance(v,dict):v={name:row for name,row in v.items() if name in cases}
    out[k]=clean(v)
   return out
  if isinstance(value,list):return [clean(v) for v in value]
  if isinstance(value,str):
   if any(path in value for path in private):return '[private validation artifact withheld]'
   return re.sub(r'(?:original|cutin|scene)[0-9]+',lambda m:m[0] if m[0] in cases or m[0]=='scene'+str(scene) else '[withheld_case]',value)
  return value
 prefixes=('report_','terminal_','evaluation_','registry_','diagnostic_','training_stress_')
 allowed={'source','champion','history','acceptance_policy','new_edit_failure','new_edit_instruction','scenario_design_hypothesis','host_protocol_revision','repair_scope'}
 items={}
 for k,item in e.get('items',{}).items():
  if k not in allowed and not k.startswith(prefixes):continue
  if item.get('case') and item['case'] not in cases:continue
  paths=[item.get('asset',{}).get('path',''),item.get('source_asset',{}).get('path','')]
  if any(any(p in path for p in private) for path in paths):continue
  items[k]=item
 e['items']=items;e=clean(e)
 if config['generalization']['mode']=='counterexample_cross_scene':
  from validation_lifecycle import verify_release
  for i,binding in enumerate(state.get('released_counterexample_manifests',[])):
   release=verify_release(binding,config)
   failures=[]
   for j,failure in enumerate(release['failed_cases']):
    details=failure['details'];failures.append({'case':failure['case']['id'],'scene':failure['case']['scene'],'reasons':failure['failure']['reasons'],'termination_reason':details.get('termination_reason'),'controls':details['controls'],'proof':failure['proof']})
    report=details.get('source_report') or details['report']
    name='report_released_'+str(i)+'_'+str(j)
    e['items'][name]={'asset':report,'case':failure['case']['id'],'facts':{'status':details.get('status'),'controls':details['controls'],'termination_reason':details.get('termination_reason'),'validation_reasons':failure['failure']['reasons']},'episode_role':'released_training_counterexample'}
    e.setdefault('diagnostic_limits',{}).setdefault('items',{})[name]={'trace_index_max':details['controls']-1}
   e['items']['released_counterexample_'+str(i)]={'asset':binding,'facts':{'role':'retired validation now training only','failed_cases':failures,'reuse_for_validation_forbidden':True}}
  e['generalization_feedback']={'policy':'Failed pools retire before disclosure; next validation uses fresh non-training scenes','validation_details_available_only_after_retirement':True}
  return e
 from blind_validation import public_feedback
 e['generalization_feedback']=public_feedback(state)
 e['items']['blind_validation_feedback']={'facts':public_feedback(state)}
 if (state.get('blind_repair') or {}).get('active'):
  e['training_only_robustness_review']=dict(require_new_causal_test=True,review=['units and frames','sampling and timing invariance','symmetry and numerical stability','risk/corridor consistency','training-scene perturbations'],equal_training_score_allowed_without_gain_claim=True,hidden_scenes_may_not_be_requested=True)
 return e
