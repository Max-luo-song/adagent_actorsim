"""Assistant-authored JSON -> isolated candidate -> registered tests -> evaluation queue."""

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

import argparse,ast,copy,datetime,hashlib,importlib.util,json,os,re,shutil,subprocess,sys
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
STORE=P26/'automated_patch_workflow_v2';BASE=ROOT/'integrations/light-intent-recovery-v12-r2';RELEASE=P26/'light_intent_recovery_v12_r2'
RUN=P26/'patch_v12_validation_r2_v1/cases/cutin551/on/B'
PYTHON=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'
class DiagnosticsPending(ValueError):pass

SAFE_METHODS=('actor_certificate','joint_certificate','rollout','terminal_stop','stop_states','admit','bounds','process','observe_conflicts','intent_pool')
def read(p):
 def bad(v):raise ValueError('Nonfinite JSON: '+v)
 return _hugsim_expand(json.loads(Path(p).read_text(),parse_constant=bad))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(v):
 if sha(v['path'])!=v['sha256']:raise ValueError('Changed artifact: '+v['path'])
def identity(s):
 if not isinstance(s,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,63}',s):raise ValueError('Invalid ID')
 return s
def pins(folder):return {str(p.relative_to(folder)):sha(p) for p in sorted([*Path(folder).rglob('*.py'),Path(folder)/'run.sh'])}
def update(round,stage,**extra):write(Path(round)/'state.json',dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
def catalog(round,candidate=None,release=None):
 platform=ROOT/'integrations/scene-platform';sys.path.insert(0,str(platform));from platform_catalog import Catalog,PROFILE,PROGRAM
 data=copy.deepcopy(read(platform/'catalog.json'));old=data['drivers'][PROFILE]
 for code,root,name,program in [(BASE,RELEASE,'intent_recovery_v12_diagnostic','scene551_native3_cutin_stop_v12')]+([(Path(candidate),Path(release),'reference_join_v13_diagnostic','scene551_native3_cutin_stop_v13')] if candidate else []):
  profile=copy.deepcopy(old);profile.update(code_dir=str(code),release_root=str(root),frozen_sources=[asset(p) for p in sorted(code.rglob('*.py'))]+[asset(code/'run.sh'),asset(root/'plan.json'),asset(root/'checks/cpu_admission.json'),asset(root/'scenario/scenario.json'),asset(root/'scenario/target_track.npz')],existing_GPU_controls=188 if code==BASE else 0,existing_GPU_anomaly=None,existing_GPU_termination='source_scene_time_window_complete' if code==BASE else None,safe_driving_certified=False)
  data['drivers'][name]=profile
  resource=copy.deepcopy(data['programs'][PROGRAM]);resource.update(driver_profile=name,resources=[asset(root/'scenario/scenario.json'),asset(root/'scenario/target_track.npz')]);data['programs'][program]=resource
 path=Path(round)/'catalog.json';write(path,data);return Catalog(path)
def capture(round):
 round=Path(round)
 if (round/'evidence.json').exists():raise FileExistsError('Round evidence already exists')
 round.mkdir(parents=True,exist_ok=True)
 report=read(RUN/'run_report.json');summary=read(RUN/'summary.json');rows=[]
 for t in report['trace']:
  r=t['road_reference'];chosen=r.get('selected_candidate');neutral=[c for c in r.get('candidates',[]) if c['passed'] and c['side']==0]
  if chosen and chosen['side']!=0 and neutral:rows.append(dict(index=t['index'],time_s=t['timestamp_s'],chosen_side=chosen['side'],chosen_score=chosen['score'],best_current_intent_score=min(c['score'] for c in neutral),current_intent_feasible_count=len(neutral)))
 evidence=dict(schema='patch_evidence_v1',baseline_version='v12',report=asset(RUN/'run_report.json'),summary=asset(RUN/'summary.json'),baseline_plan=asset(RELEASE/'plan.json'),baseline_code=pins(BASE),
  observations=dict(controls=summary['controls'],status=summary['status'],stopped_seconds=summary['stopped_seconds'],progress_m=summary['progress_along_initial_forward_m'],final_motion_state=report['trace'][-1]['state_after'],bypass_despite_feasible_current_intent=rows),
  interpretation='Measured summaries only; causal explanation must be authored by acting LLM; lateral offset alone does not establish unsafe driving')
 write(round/'evidence.json',evidence);catalog(round)
 write(round/'llm_request.json',dict(schema='patch_llm_request_v1',decision_provider='current_conversation_assistant',external_model_API_called=False,user_diagnosis_required=False,
  input=asset(round/'evidence.json'),required_outputs=['edit_instruction.json','failure_reason.json','patch_instruction.json'],
  rules=['Use frozen v12 only; create isolated v13','Separate observed facts, hypotheses and supported mechanisms','May request diagnostics rather than patch','Never relax safety checks or change tests/criteria to obtain acceptance','Acceptance needs causal tests plus matched closed-loop and nonregression evidence']))
 update(round,'awaiting_assistant_decision');return evidence['observations']
def path_under(folder,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Patch path escapes candidate')
 result=Path(folder)/p
 if result.is_symlink():raise ValueError('Symlink target not allowed')
 return result

def safety_equal(before,after):
 def methods(s):
  tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointReference')
  return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
 a,b=methods(before),methods(after)
 for name in SAFE_METHODS:
  if a[name]!=b[name]:raise ValueError('Protected safety method changed: '+name)
 def policy(s):
  tree=ast.parse(s);n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='POLICY' for x in n.targets));return {k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
 a,b=policy(before),policy(after)
 if any(b.get(k)!=v for k,v in a.items()) or set(b)-set(a)-{'intent_recovery_hold_s'}:raise ValueError('Original physical/safety policy changed')

def memory_equal(before,after):
 class StripState(ast.NodeTransformer):
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr in {'intent_clear_since','intent_last_clock','intent_recovery_audit','intent_clear_observations'} for t in node.targets):return None
   return self.generic_visit(node)
 if ast.dump(StripState().visit(ast.parse(before)))!=ast.dump(StripState().visit(ast.parse(after))):raise ValueError('Obstacle memory geometry/lifecycle changed beyond recovery-state reset')

def decisions(round):
 round=Path(round)
 if (round/'diagnostic_gate.json').exists() and not read(round/'diagnostic_gate.json')['passed']:raise ValueError('This proposal was rejected by the assistant diagnostic gate')
 e=read(round/'evidence.json');f=read(round/'failure_reason.json');edit=read(round/'edit_instruction.json')
 for key in ['report','summary','baseline_plan']:verify(e[key])
 if pins(BASE)!=e['baseline_code']:raise ValueError('Frozen v12 sources changed')
 if f.get('schema')!='failure_reason_v1' or f.get('author',{}).get('role')!='acting_llm_assistant':raise ValueError('Assistant analysis is required; executor does not invent causes')
 if f.get('evidence')!=asset(round/'evidence.json'):raise ValueError('Decision/evidence chain mismatch')
 if f.get('decision') in ('diagnostics_required','no_supported_failure'):
  update(round,'awaiting_diagnostics' if f['decision']=='diagnostics_required' else 'no_patch_supported',promoted=False,analysis=asset(round/'failure_reason.json'));raise DiagnosticsPending('No patch application: '+f['decision'])
 p=read(round/'patch_instruction.json')
 if p.get('failure_reason')!=asset(round/'failure_reason.json'):raise ValueError('Decision/evidence chain mismatch')
 if p.get('schema')!='patch_instruction_v1' or p.get('base_version')!='v12' or p.get('candidate_version')!='v13':raise ValueError('Version contract changed')
 if f.get('decision')!='candidate_patch':raise ValueError('Diagnostic request must be resolved before automatic patch application')
 if p.get('acceptance')!=dict(causal_tests_required=True,matched_closed_loop_required=True,normal_scene_nonregression_required=True,cross_scene_nonregression_required=True):raise ValueError('Acceptance gates cannot be weakened')
 for proof in f.get('evidence_links',[]):verify(proof)
 sys.path.insert(0,str(ROOT/'integrations/scene-platform'));from platform_catalog import Catalog
 from platform_contracts import compile_task
 c=Catalog(round/'catalog.json');c.verify(edit['simulation']['profile']);compiled=compile_task(edit,c)
 if edit['simulation']['profile']!='intent_recovery_v12_diagnostic' or edit['scene_id']!='551':raise ValueError('Edit task does not match measured v12 episode')
 if edit['simulation']['control_steps']!=188 or len(edit['edits'])!=1 or edit['edits'][0]['program_id']!='scene551_native3_cutin_stop_v12':raise ValueError('Reproduction scenario changed')
 write(round/'compiled_edit.json',compiled);return e,f,p,edit

def apply(round):
 round=Path(round);e,f,p,edit=decisions(round);candidate=ROOT/'integrations'/identity(p['candidate_directory']);namespace=identity(p['output_namespace']);output=P26/namespace
 receipt=round/'apply_receipt.json'
 if receipt.exists():verify_candidate(round);return read(receipt)
 if candidate.exists() or output.exists():raise FileExistsError('Candidate/output already exists without this round receipt')
 if not re.fullmatch(r'light-reference-join-v13(?:-r[0-9]+)?',candidate.name) or namespace!=candidate.name.replace('-','_'):raise ValueError('This release adapter admits only isolated v13 revision names')
 operations=p.get('operations',[])
 if not operations:raise ValueError('Empty patch')
 allowed={'joint_safety.py','reference_join.py','cpu_check.py','pipeline.py','compare_native.py','tests/test_reference_join.py'}
 for op in operations:
  if op['path'] not in allowed or op['op'] not in ('replace_exact','create_file'):raise ValueError('Unregistered source edit')
  path_under(candidate,op['path'])
 shutil.copytree(BASE,candidate,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 replacements={'light_intent_recovery_v12_r2':namespace,'intent_recovery_v12_code':'reference_join_v13_code','joint_intent_recovery_v12':'joint_reference_join_v13'}
 for file in [*candidate.rglob('*.py'),candidate/'run.sh']:
  s=file.read_text()
  for a,b in replacements.items():s=s.replace(a,b)
  file.write_text(s)
 before=(candidate/'joint_safety.py').read_text();memory_before=(candidate/'risk_memory.py').read_text()
 for op in operations:
  target=path_under(candidate,op['path'])
  if op['op']=='create_file':
   if target.exists():raise ValueError('Create target already exists')
   target.parent.mkdir(parents=True,exist_ok=True);target.write_text(op['content'])
  else:
   s=target.read_text()
   if s.count(op['before'])!=1:raise ValueError('Patch context not unique: '+op['path'])
   target.write_text(s.replace(op['before'],op['after']))
 for file in candidate.rglob('*.py'):ast.parse(file.read_text(),filename=str(file))
 safety_equal(before,(candidate/'joint_safety.py').read_text());memory_equal(memory_before,(candidate/'risk_memory.py').read_text())
 result=dict(candidate=str(candidate),output=str(output),candidate_code=pins(candidate),source_code=e['baseline_code'],
  instruction=asset(round/'patch_instruction.json'),analysis=asset(round/'failure_reason.json'),edit=asset(round/'edit_instruction.json'),safety_AST_unchanged=True,source_v12_unchanged=pins(BASE)==e['baseline_code'])
 write(receipt,result);update(round,'candidate_applied',candidate=str(candidate),promoted=False);return result

def verify_candidate(round):
 r=read(Path(round)/'apply_receipt.json');verify(r['instruction']);verify(r['analysis']);verify(r['edit'])
 analysis=read(r['analysis']['path']);verify(analysis['evidence']);evidence=read(analysis['evidence']['path'])
 for key in ('report','summary','baseline_plan'):verify(evidence[key])
 if pins(Path(r['candidate']))!=r['candidate_code'] or pins(BASE)!=r['source_code']:raise ValueError('Candidate or baseline changed after application')
 return r

def run_registered(round,name,cmd,reuse=True):
 round=Path(round);verify_candidate(round)
 if (round/'diagnostic_gate.json').exists() and not read(round/'diagnostic_gate.json')['passed']:raise ValueError('Rejected proposal cannot continue')
 folder=round/'execution';folder.mkdir(exist_ok=True);log=folder/(name+'.log');receipt=folder/(name+'.json')
 if reuse and receipt.exists():
  previous=read(receipt)
  if previous['returncode']==0:
   if previous['command']!=cmd:raise ValueError('Registered command changed; create a new round')
   verify(previous['log']);print(name,'reused verified receipt',flush=True);return
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for k in ['PYTHONPATH','PYTHONHOME']:env.pop(k,None)
 with log.open('a') as stream:
  code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
 verify_candidate(round);write(folder/(name+'.json'),dict(stage=name,returncode=code,log=asset(log),command=cmd,executor_at_recording=asset(LOCAL/'workflow.py')))
 if code:update(round,'verification_failed',failed_stage=name,log=str(log),promoted=False);raise RuntimeError(name+' failed; inspect '+str(log))
 print(name,'passed',flush=True)

def prepare_validation(round):
 from validation_builder import prepare_validation as prepare
 return prepare(round)

def cycle(round):
 round=Path(round);r=apply(round);candidate=Path(r['candidate'])
 if (round/'state.json').exists() and (round/'validation_harness.json').exists():
  previous=read(round/'state.json')
  if previous['stage'] in ('awaiting_gpu_validation','pending_evidence','validated_on_frozen_matrix','rejected'):
   h=read(round/'validation_harness.json')
   if pins(Path(h['path']))!=h['code']:raise ValueError('Validation harness changed')
   return previous
 update(round,'cpu_verification_running',promoted=False)
 run_registered(round,'candidate_cpu',[str(PYTHON),'-u','-B',str(candidate/'cpu_check.py')])
 run_registered(round,'candidate_prepare',['bash',str(candidate/'run.sh'),'prepare'])
 run_registered(round,'candidate_check',['bash',str(candidate/'run.sh'),'check'])
 v=prepare_validation(round)
 update(round,'awaiting_gpu_validation',candidate_version='v13',CPU_verified=True,global_patch_validated=False,promoted=False,validation=v,
  next_command=['bash',str(Path(v['path'])/'run.sh'),'full'],decision_provider='current_conversation_assistant',external_model_API_called=False)
 return read(round/'state.json')

def incremental_target(round,h,evaluation):
 current=evaluation['details'].get('cutin551',{}).get('candidate_v13',evaluation['details'].get('cutin551',{}).get('on'))
 if current is None:return dict(status='pending',accepted=False,reason='No verified v13 target closed loop yet')
 if current.get('score') is None or current.get('status')!='completed':return dict(status='invalid',accepted=False,reason='v13 target is incomplete or unscored')
 matrix=read(Path(h['output'])/'matrix.json');work=Path(matrix['cases']['cutin551']['arms']['on'])
 for name in ('scene_manifest.json','initial_context.json','map/map.json','scenario/scenario.json','scenario/target_track.npz'):
  if sha(RELEASE/name)!=sha(work/name):return dict(status='invalid',accepted=False,reason='v12/v13 exogenous asset mismatch: '+name)
 before=read(RUN/'summary.json');after=read(work/'B/summary.json')
 old=before['native_scores'];new=after['native_scores'];keys=set(old['comparison_key'])-{'actor_values_sha256'}
 if any(old['comparison_key'][k]!=new['comparison_key'].get(k) for k in keys):return dict(status='invalid',accepted=False,reason='v12/v13 native scoring keys differ')
 import numpy as np
 with np.load(RUN/'actor_tracks.npz') as a,np.load(work/'B/actor_tracks.npz') as b:
  if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):return dict(status='invalid',accepted=False,reason='v12/v13 actor worlds differ')
 scores=old['scores']['executed_path'];hd=current['score']['hdscore']-scores['hdscore']['value'];pdms=current['score']['pdms']-scores['pdms']['value']
 reasons=[]
 if current['collision'] or current['map_departure'] or current['unresolved_stop'] or current['status']!='completed':reasons.append('v13_target_safety_or_completion_failure')
 if hd<-.01 or pdms<-.01:reasons.append('v13_target_score_regression_against_v12')
 if current['progress_m']<before['progress_along_initial_forward_m']*.95 or current['stopped_s']>before['stopped_seconds']+1.:reasons.append('v13_target_progress_or_stop_regression_against_v12')
 return dict(status='failed' if reasons else 'improved' if hd>1e-9 else 'nonregression_passed',accepted=not reasons,hd_gain=hd,pdms_gain=pdms,reasons=reasons,
  baseline=asset(RUN/'summary.json'),candidate=asset(work/'B/summary.json'),scope='v12 to v13 target increment only; not full patch-off comparison')

def evaluate(round):
 round=Path(round);verify_candidate(round)
 if (round/'diagnostic_gate.json').exists() and not read(round/'diagnostic_gate.json')['passed']:
  result=dict(status='rejected',promoted=False,reason='Assistant diagnostic gate failed',gate=asset(round/'diagnostic_gate.json'));write(round/'acceptance.json',result);return result
 h=read(round/'validation_harness.json');harness=Path(h['path'])
 if pins(harness)!=h['code']:raise ValueError('Validation code changed')
 run_registered(round,'validation_evaluation',['bash',str(harness/'run.sh'),'summarize'],reuse=False)
 e=read(Path(h['output'])/'evaluation.json');increment=incremental_target(round,h,e);local=all(e['results'].get(k,{}).get('passed') for k in ['original551','cutin551'])
 result=dict(schema='patch_acceptance_v1',candidate_version='v13',evaluation=asset(Path(h['output'])/'evaluation.json'),
   local_pairs_passed=local,global_patch_validated=e['global_patch_validated'] and increment['accepted'],promoted=e['global_patch_validated'] and increment['accepted'],incremental_v12_comparison=increment,
   status='validated_on_frozen_matrix' if e['global_patch_validated'] and increment['accepted'] else 'rejected' if e['status']=='failed' or increment['status']=='failed' else 'pending_evidence',
   missing_cases=e['missing_cases'],scope='engineering matrix only; no universal or formal certification')
 write(round/'acceptance.json',result);update(round,result['status'],acceptance=result,promoted=result['promoted'])
 write(round/'llm_review_request.json',dict(schema='patch_review_request_v1',decision_provider='current_conversation_assistant',external_model_API_called=False,evaluation=result['evaluation'],original_analysis=asset(round/'failure_reason.json'),next='inspect new evidence and diagnose failures; do not request user diagnosis or claim missing-scene acceptance'))
 return result

def gpu(round):
 round=Path(round);verify_candidate(round);h=read(round/'validation_harness.json')
 if pins(Path(h['path']))!=h['code']:raise ValueError('Validation harness changed')
 if (round/'diagnostic_gate.json').exists() and not read(round/'diagnostic_gate.json')['passed']:raise ValueError('Rejected candidate')
 import torch
 if not torch.cuda.is_available():
  update(round,'awaiting_gpu_validation',candidate_version='v13',CPU_verified=True,promoted=False,global_patch_validated=False,GPU_executed=False,reason='CUDA unavailable',next_command=['bash',str(LOCAL/'run.sh'),'gpu']);return read(round/'state.json')
 run_registered(round,'gpu_validation',['bash',str(Path(h['path'])/'run.sh'),'full'],reuse=False)
 return evaluate(round)

def dispatch(a):
 if a.stage=='capture':r=capture(a.round)
 elif a.stage=='apply':r=apply(a.round)
 elif a.stage=='cycle':r=cycle(a.round)
 elif a.stage=='status':r=read(a.round/'state.json')
 elif a.stage=='evaluate':r=evaluate(a.round)
 elif a.stage=='gpu':r=gpu(a.round)
 else:
  _,_,_,edit=decisions(a.round)
  if a.stage=='compile-edit':r=read(a.round/'compiled_edit.json')
  else:
   from platform_runner import Runner
   from platform_catalog import Catalog
   runner=Runner(a.round/'scene_jobs',Catalog(a.round/'catalog.json'))
   if a.stage=='prepare-edits':
    patched=read(a.round/'patched_edit_instruction.json')
    r=dict(baseline_job=str(runner.prepare(edit,'v12_failure_reproduction')),candidate_job=str(runner.prepare(patched,'v13_candidate_instruction')),controls_executed=False)
    write(a.round/'scene_instruction_receipts.json',r)
   elif a.stage=='run-patched-edit':
    verify_candidate(a.round);patched=read(a.round/'patched_edit_instruction.json');r=runner.run(patched,'v13_candidate_instruction')
   else:r=runner.run(edit,'v12_failure_reproduction')
 return r

def main():
 import fcntl
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['capture','apply','cycle','status','compile-edit','run-edit','run-patched-edit','prepare-edits','evaluate','gpu']);ap.add_argument('--round',type=Path,default=STORE/'rounds/v12_to_v13_reference_join_r2');a=ap.parse_args()
 if a.stage in ('capture','status'):r=dispatch(a)
 else:
  with (a.round/'execution.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   try:r=dispatch(a)
   except DiagnosticsPending:r=read(a.round/'state.json')
 print(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
