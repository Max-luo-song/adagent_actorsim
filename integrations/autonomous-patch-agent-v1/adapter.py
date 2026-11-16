"""Registered HUGSIM adapter. Commands and evaluation stay host-owned."""

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

import ast,sys
pass  # Path is provided by the portable boundary above
from core import AgentError,LOCAL,ROOT,P26,asset,code_pins,read,sha,verify,verify_pins,write
import builder
from validation import actual_metrics
class HUGSIMAdapter:
 simulation_only=False
 def bootstrap(self,config):
  seed=config['seed'];matrix=Path(seed['matrix']);verify_pins(read(matrix/'matrix.json')['code_sha256'])
  admission=read(Path(seed['release'])/'checks/cpu_admission.json');verify_pins(admission['code_sha256'])
  if not admission['passed']:raise AgentError('Seed CPU admission failed')
  registered=read(matrix/'matrix.json')
  verify(registered['source_release']);verify(registered['source_patch_cpu'])
  if registered['source_patch_cpu']!=asset(Path(seed['release'])/'checks/cpu_admission.json'):raise AgentError('Seed CPU/GPU release linkage differs')
  source_plan=read(registered['source_release']['path'])
  if source_plan['execution_horizon_v17_code']!=code_pins(seed['code']):raise AgentError('Seed code differs from executed release')
  acceptance=read(seed['acceptance'])
  if not acceptance.get('working_baseline_improved') or not acceptance.get('path_comparison',{}).get('passed'):raise AgentError('Seed is not an accepted local working result')
  verify(acceptance['evaluation'])
  metrics=actual_metrics(matrix);reference=actual_metrics(config['reference_matrix'])
  for item in [*metrics.values(),*reference.values()]:
   if item['status']!='completed' or item['controls']!=188 or item['observations']!=189 or item['collision'] or item['map_departure'] or item['unresolved_stop']:raise AgentError('Unhealthy completed bootstrap pair')
  return dict(version='seed_v17_r2',code=seed['code'],release=seed['release'],release_name=Path(seed['release']).name,matrix=seed['matrix'],code_pins=code_pins(seed['code']),metrics=metrics,reference_metrics=reference,seed_acceptance=asset(seed['acceptance']))
 def build(self,round_dir,rid,champion,proposal,config):return builder.build(round_dir,rid,champion,proposal,config)
 def attempt(self,round_dir,bundle,number):return builder.attempt(round_dir,bundle,number)
 def stages(self,round_dir,bundle,attempt,config):
  rd=Path(round_dir);c=Path(bundle['code']);release=Path(bundle['release']);h=Path(attempt['harness']);out=Path(attempt['output']);a=Path(attempt['root']);checks=rd/'checks';checks.mkdir(exist_ok=True)
  base={**bundle['candidate_pins'],**attempt['harness_pins']};py=str(ROOT/'code/HUGSIM/.pixi/envs/default/bin/python');worker=lambda file,*args:[py,'-u','-B',str(file),*map(str,args)]
  def stage(name,command,outputs=(),deps=(),gpu=False):
   inputs=dict(base)
   return dict(name=name,command=command,outputs=list(map(str,outputs)),inputs=inputs,deps=list(map(str,deps)),timeout=config['budgets']['gpu_stage_seconds'] if gpu else config['budgets']['cpu_stage_seconds'],gpu=gpu)
  baseline=checks/'causal_parent.json';candidate=checks/'causal_candidate.json';extra=checks/'additional.json';preflight=checks/'smpl.json';replay=release/'checks/v13_matched_replay.json';cpu=release/'checks/cpu_admission.json';hcpu=out/'checks/cpu_tests.json'
  parent_cmd=worker(LOCAL/'test_runner.py','--code',bundle['parent_fixture'],'--test',bundle['new_test'],'--output',baseline)
  causal=worker(LOCAL/'test_runner.py','--code',c,'--output',candidate)
  for p in sorted((c/'tests').glob('test_agent_*.py')):causal+=['--test',str(p)]
  extra_cmd=worker(LOCAL/'test_runner.py','--code',c,'--output',extra)
  for p in bundle['extra_tests']:extra_cmd+=['--test',p]
  fixed=[stage('preflight',worker(LOCAL/'probe.py','--code',c,'--output',preflight),[preflight]),stage('causal_parent',parent_cmd,[baseline],list(bundle['parent_fixture_pins'])),stage('causal_candidate',causal,[candidate]),stage('registered_cpu',worker(c/'cpu_check.py'),[cpu]),stage('additional',extra_cmd,[extra],bundle['extra_tests']),stage('matched_replay',worker(bundle['replay']),[replay],[bundle['replay']])]
  fixed += [stage('harness_cpu',['bash',str(h/'run.sh'),'cpu'],[hcpu]),stage('candidate_prepare',['bash',str(c/'run.sh'),'prepare'],[release/'plan.json'],[cpu]),stage('candidate_check',['bash',str(c/'run.sh'),'check'],[release/'plan.json'],[cpu,release/'plan.json']),stage('harness_prepare',['bash',str(h/'run.sh'),'prepare'],[out/'matrix.json'],[cpu,hcpu,release/'plan.json']),stage('harness_check',['bash',str(h/'run.sh'),'check'],[out/'matrix.json'],[out/'matrix.json']),stage('gpu',['bash',str(h/'run.sh'),'full'],[out/'evaluation.json'],[out/'matrix.json'],gpu=True)]
  return fixed
 def validate_stage(self,name,round_dir,bundle,attempt):
  checks=Path(round_dir)/'checks';release=Path(bundle['release']);out=Path(attempt['output'])
  if name=='causal_parent':
   r=read(checks/'causal_parent.json')
   if r['tests']<1 or r['failures']<1 or r['errors'] or r['skipped'] or r['expected_failures'] or r['unexpected_successes']:raise AgentError('New causal test must fail by assertion on unchanged parent, without import errors/skips')
   if r['controller']!=asset(Path(bundle['parent_fixture'])/'joint_safety.py'):raise AgentError('Parent test target mismatch')
  elif name in ('causal_candidate','additional'):
   r=read(checks/('causal_candidate.json' if name=='causal_candidate' else 'additional.json'))
   if not r['passed'] or r['tests']<1 or r['skipped'] or r['expected_failures'] or r['unexpected_successes']:raise AgentError('Candidate causal/registered regression did not pass')
   if r['controller']!=asset(Path(bundle['code'])/'joint_safety.py'):raise AgentError('Candidate test target mismatch')
  elif name=='registered_cpu':
   r=read(release/'checks/cpu_admission.json')
   if not r['passed'] or r['code_sha256']!=bundle['candidate_pins'] or r['tests']<76:raise AgentError('Full registered CPU admission missing')
  elif name=='matched_replay':
   r=read(release/'checks/v13_matched_replay.json')
   if not r['passed'] or len(r['rows'])!=40:raise AgentError('Matched-observation replay failed')
  elif name=='preflight':
   r=read(checks/'smpl.json')
   if not r['passed'] or not r['actual_framework_template_constructed'] or not r['existing_pickle_bypassed']:raise AgentError('Actual SMPL template preflight required')
  elif name=='harness_prepare':
   matrix=read(out/'matrix.json');cases={}
   for case,arm in [('original551','A'),('cutin551','B')]:
    work=Path(matrix['cases'][case]['arms']['on']);contract=read(work/'contract.json')
    cases[case]=dict(scene_manifest=asset(work/'scene_manifest.json'),scenario=asset(work/'scenario/scenario.json'),target_track=asset(work/'scenario/target_track.npz') if case=='cutin551' else None,contract=asset(work/'contract.json'),controls=contract['controls'],fresh_history=True,source_clock=contract['initial_context'],actual_actor_program='registered original actors' if case=='original551' else 'registered canonical original-vehicle cut-in')
   write(Path(round_dir)/'compiled_edit_instruction.json',dict(schema='registered_edit_execution_v1',cases=cases,model_cannot_change_scenario=True,execution_commands_owned_by_host=True))
  elif name=='harness_cpu':
   r=read(out/'checks/cpu_tests.json')
   if not r['passed'] or r['tests']<17 or r['code_sha256']!=attempt['harness_pins']:raise AgentError('Harness CPU proof failed')
  verify_pins(bundle['candidate_pins']);verify_pins(attempt['harness_pins'])
 def results(self,bundle,attempt):return actual_metrics(attempt['output']),read(Path(attempt['output'])/'evaluation.json')
 def classify_failure(self,stage,receipt):
  if receipt['status'] in ('interrupted','timeout','cancelled'):return 'interrupted'
  if stage['gpu']:
   tail=Path(receipt['log']['path']).read_text(errors='replace')[-12000:] if receipt.get('log') else ''
   if receipt.get('returncode')==75 or 'GPU lock busy' in tail:return 'gpu_busy'
   if 'CUDA unavailable' in tail:return 'gpu_unavailable'
   return 'infrastructure'
  return 'offline_rejection'
