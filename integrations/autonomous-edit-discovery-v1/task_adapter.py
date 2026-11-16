"""Concrete v7 adapter for a newly generated 551 edit; old task regresses afresh."""

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

import copy,sys
pass  # Path is provided by the portable boundary above
from common_flow import LOCAL as FLOW,OUT,read as flow_read,verify_pins as flow_verify,metrics,healthy,verify as flow_verify_asset
sys.path.insert(0,str(FLOW.parents[0]/'autonomous-patch-agent-v7'))
from adapter import HUGSIMAdapter
from core import ROOT,P26,asset,read,write,verify_pins,code_pins
from validation import acceptance

class NewEditAdapter(HUGSIMAdapter):
 def __init__(self,task):self.task=copy.deepcopy(task);flow_verify_asset(task['task_asset']);self.file=Path(task['task_file']);self.spec=read(self.file);flow_verify(self.spec['host_pins'])
 def bootstrap(self,config):return copy.deepcopy(read(self.spec['admission'])['champion'])
 def registered_pool(self,config,champion):return copy.deepcopy(read(self.spec['admission'])['registry'])
 def attempt(self,round_dir,bundle,number):
  root=Path(round_dir)/'attempts'/('a%02d'%number);root.mkdir(parents=True,exist_ok=True)
  out=root/'matrix';r=dict(number=number,root=str(root),harness=str(ROOT/'integrations/patch-v17-validation-r2-v1'),output=str(out),harness_pins=code_pins(ROOT/'integrations/patch-v17-validation-r2-v1'))
  write(root/'bundle.json',r);return r
 def stages(self,rd,bundle,attempt,config):
  # Preserve causal, full registered policy, replay, worker and native IPOPT checks.
  old=super().stages(rd,bundle,attempt,config)
  names={'preflight','causal_parent','causal_candidate','registered_cpu','additional','matched_replay','candidate_prepare','candidate_check'}
  stages=[s for s in old if s['name'] in names]
  root=Path(attempt['root']);policy=root/'policy.json';write(policy,dict(version=bundle['round_id'],code=bundle['code'],release=bundle['release'],release_name=bundle['release_name'],code_pins=bundle['candidate_pins']))
  args=['--task',str(self.file),'--policy',str(policy),'--output',attempt['output']]
  deps={**bundle['candidate_pins'],**self.spec['host_pins']}
  legacy_cpu=Path(attempt['root'])/'legacy_harness_cpu'
  stages.append(dict(name='legacy_harness_cpu',command=['bash',str(Path(attempt['harness'])/'run.sh'),'cpu',str(legacy_cpu)],outputs=[str(legacy_cpu/'checks/cpu_tests.json')],inputs={**deps,**attempt['harness_pins']},deps=[],timeout=config['budgets']['cpu_stage_seconds'],gpu=False))
  stages.append(dict(name='new_task_prepare',command=['bash',str(FLOW/'run.sh'),'_prepare',*args],outputs=[str(Path(attempt['output'])/'prepared.json')],inputs=deps,deps=[],timeout=config['budgets']['cpu_stage_seconds'],gpu=False))
  stages.append(dict(name='gpu',command=['bash',str(FLOW/'run.sh'),'_gpu',*args],outputs=[str(Path(attempt['output'])/'evaluation.json')],inputs=deps,deps=[str(Path(attempt['output'])/'prepared.json')],timeout=config['budgets']['gpu_stage_seconds'],gpu=True))
  return stages
 def validate_stage(self,name,rd,bundle,attempt):
  if name=='legacy_harness_cpu':
   proof=read(Path(attempt['root'])/'legacy_harness_cpu/checks/cpu_tests.json')
   if not proof['passed'] or proof['tests']<17 or proof['code_sha256']!=attempt['harness_pins']:raise ValueError('Existing registered harness gates failed')
   return
  if name=='new_task_prepare':
   p=read(Path(attempt['output'])/'prepared.json');flow_verify(p['frozen_inputs']);flow_verify(self.spec['host_pins']);return
  return super().validate_stage(name,rd,bundle,attempt)
 def results(self,bundle,attempt):
  out=Path(attempt['output']);r=read(out/'evaluation.json');flow_verify(self.spec['host_pins']);return r['metrics'],r
 def acceptance(self,candidate,champion,reference,policy,strict):
  result=acceptance(candidate,champion,reference,policy,strict)
  legacy=strict['legacy_metrics'];old=read(self.spec['source_state'])['champion']['metrics']
  if any(not healthy(legacy[c]) for c in legacy):result['reasons'].append('legacy_task_incomplete_or_failed')
  for c in legacy:
   n,b=legacy[c],old[c]
   if n['score']['hdscore']<b['score']['hdscore']-policy['numeric_tolerance'] or n['score']['pdms']<b['score']['pdms']-policy['numeric_tolerance']:result['reasons'].append('legacy_task_score_regression')
   if n['progress_m']<b['progress_m']*policy['min_progress_ratio'] or n['stopped_s']>b['stopped_s']+policy['max_extra_stopped_seconds']:result['reasons'].append('legacy_task_progress_regression')
   if any(n[k]>b[k]+policy['lateral_tolerance_m'] for k in ('final_abs_lateral_m','peak_abs_lateral_m')):result['reasons'].append('legacy_task_lateral_regression')
  # Research frontier remains protected on its own original task, not transplanted across edits.
  frontier=read(self.spec['source_registry']);best=max(v['metrics']['cutin551']['score']['hdscore'] for v in frontier.values() if healthy(v['metrics']['cutin551']))
  if legacy['cutin551']['score']['hdscore']<best-policy['numeric_tolerance']:result['reasons'].append('legacy_high_score_frontier_gain_lost')
  result['accepted']=not result['reasons'];result['scope']='new edited task plus actual legacy regression; held-out scenes still pending';return result
 def collect_evidence(self,config,champion,history,latest):
  from evidence import collect
  evidence=collect(config,champion,history,latest)
  evidence['items']['new_edit_failure']={'asset':asset(self.spec['failure_reason']),'facts':read(self.spec['failure_reason'])}
  evidence['items']['new_edit_instruction']={'asset':asset(self.file),'facts':self.spec['edit']}
  return evidence
 def activate_task(self,task,carried,archives,budget):
  admission=read(self.spec['admission']);champ=admission['champion']
  if champ['code_pins']!=carried['code_pins']:raise ValueError('New baseline belongs to a different inherited policy')
  return dict(task_id=task['id'],admission=asset(self.spec['admission']),champion=champ,reference_metrics=champ['reference_metrics'],registry=admission['registry'],target_case='cutin551',validated=True)
 def verify_task_admission(self,task,activation):
  a=read(activation['admission']['path']);return a['qualified'] and a['scene']==task['scene'] and a['edit']==task['edit'] and all(healthy(m) for m in a['champion']['metrics'].values())
 def retention_check(self,bundle,attempt,archives):
  # Any additional archived tasks must have real registered regression episodes.
  r=read(Path(attempt['output'])/'evaluation.json')
  return dict(passed=not archives,candidate_pins=bundle['candidate_pins'],tasks={})

def make_adapter(task,config,checkpoint):return NewEditAdapter(task)
