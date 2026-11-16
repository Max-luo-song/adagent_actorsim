"""New-edit discovery -> repeated inherited-policy probes -> real v7 LLM round."""

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

import argparse,copy,fcntl,os,subprocess,sys,time
pass  # Path is provided by the portable boundary above
from common_flow import *
PYTHON=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'


def frozen_host():
 return {str(p):sha(p) for folder in (LOCAL,DISCOVERY,AGENT) for p in folder.rglob('*') if p.suffix in ('.py','.sh','.json','.txt') and '__pycache__' not in p.parts}


def status(stage,**extra):write(OUT/'status.json',dict(stage=stage,updated_epoch=time.time(),**extra))


def initialize(extra_wall,extra_gpu):
 from migration import initialize as migrate
 return migrate(frozen_host(),dict(add_wall_seconds=extra_wall,add_gpu_attempts=extra_gpu))


def remaining(m):
 ledger=read(OUT/'ledger.json')
 if ledger['started_epoch'] is None:ledger['started_epoch']=time.time();write(OUT/'ledger.json',ledger)
 value=m['explicit_budget_extension']['wall_seconds']-(time.time()-ledger['started_epoch'])
 if value<=0:
  sys.path.insert(0,str(AGENT));from core import BudgetExceeded
  raise BudgetExceeded('Supplemental mission wall budget exhausted; counters retained')
 return value


def probe(m,taskfile,policyfile,work,case):
 work=Path(work);arm='A' if case=='original551' else 'B'
 if (work/'active_retry.json').exists():
  retry=read(work/'active_retry.json');verify(retry['task']);verify(retry['policy'])
  if retry['task']['path']!=str(Path(taskfile).resolve()) or retry['policy']['path']!=str(Path(policyfile).resolve()) or retry['case']!=case:raise RuntimeError('Retry binding changed')
  for receipt in retry['preserved_failure'].values():verify(receipt)
  next_work=Path(retry['work']).resolve()
  if not next_work.is_relative_to((work/'infra_retries').resolve()):raise RuntimeError('Retry path escapes its original episode')
  work=next_work
 command=[str(PYTHON),'-u','-B',str(LOCAL/'probe_worker.py'),'prepare','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case]
 subprocess.run(command,cwd=ROOT,check=True,env=native_environment())
 if (work/'completion.json').exists():return metrics(work,arm)
 command[4]='execute' # python, -u, -B, script, stage
 job=work/'dispatch.json'
 if job.exists():
  owner=read(job)
  sys.path.insert(0,str(DISCOVERY));from service import alive
  if alive(owner):
   status('observing_existing_probe',case=case,work=str(work))
   while alive(owner) and remaining(m)>0:time.sleep(2)
   if (work/'completion.json').exists():return metrics(work,arm)
   raise RuntimeError('Interrupted probe retained: '+str(work))
  raise RuntimeError('Dead probe without completion retained: '+str(work))
 if (work/arm).exists():raise RuntimeError('Partial episode retained; no automatic action replay: '+str(work))
 while True:
  remaining(m)
  if (OUT/'STOP').exists():raise RuntimeError('Operator stop requested')
  ledger=read(OUT/'ledger.json')
  if ledger['gpu_attempts']>=m['explicit_budget_extension']['gpu_attempts']:
   sys.path.insert(0,str(AGENT));from core import BudgetExceeded
   raise BudgetExceeded('Supplemental discovery GPU budget exhausted')
  attempt=ledger['gpu_attempts'];log=work/('gpu_attempt_%03d.log'%attempt)
  args=['bash',str(LOCAL/'run.sh'),'_probe','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case]
  with log.open('x') as stream:
   proc=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=native_environment())
   sys.path.insert(0,str(DISCOVERY));from service import identity
   write(job,dict(pid=proc.pid,identity=identity(proc.pid),case=case,log=str(log)))
   ledger['gpu_attempts']+=1;write(OUT/'ledger.json',ledger)
   while proc.poll() is None:
    if (OUT/'STOP').exists() or remaining(m)<=0:
     import signal
     os.killpg(proc.pid,signal.SIGTERM);proc.wait(10);raise RuntimeError('Probe cancelled; partial artifacts retained')
    time.sleep(1)
  if proc.returncode==75 and log.read_text().strip()=='GPU lock busy':
   # No controls ran. Preserve the attempt log without spending a GPU episode.
   ledger['gpu_attempts']-=1;write(OUT/'ledger.json',ledger)
   log.rename(log.with_name(log.name+'.busy_'+str(time.time_ns())));job.unlink()
   status('waiting_gpu',case=case);time.sleep(15);continue
  if proc.returncode:raise RuntimeError('Probe failed: '+str(log))
  return metrics(work,arm)


def task_prepare(m,entry):
 from importlib import util
 # This branch runs in a short separate process so v7 and old evidence modules cannot shadow each other.
 sys.path.insert(0,str(DISCOVERY));from episode import prepare as prepare_episode
 stable=read(m['policies']['stable']);folder=OUT/'tasks'/entry['id'];folder.mkdir(parents=True,exist_ok=True)
 taskfile=folder/'task.json'
 if taskfile.exists():return taskfile
 baseline=Path(stable['matrix'])/'cases/original551/on'
 spec=dict(schema='new_edit_task_v1',id='new_'+entry['id'],scene=551,edit=dict(id=entry['id'],scene=551,program='registered_native_action_sequence_v1',parameters={k:v for k,v in entry.items() if k!='id'}),
           baseline_root=str(baseline),source_contracts={c:stable['metrics'][c]['contract']['path'] for c in stable['metrics']},
           edit_assets=str(folder/'edit'),admission=str(folder/'admission.json'),failure_reason=str(folder/'failure_reason.json'),
           source_state=str(OUT/'source_state.json'),source_registry=str(OUT/'source_registry.json'),host_pins=m['host_pins'])
 staging=folder/'edit_work'
 if not staging.exists():
  prepare_action_template(spec['edit'],stable,staging)
  import action_scenario
  from action_program import validate_phases,PROGRAM
  if set(spec['edit']['parameters'])!={'phases'}:raise ValueError('Only multi-stage action programs may be generated')
  original=copy.deepcopy(action_scenario.CONFIG)
  action_scenario.CONFIG=copy.deepcopy(original)
  for key in ('lane_change_duration_s','actor_deceleration_duration_s','target_lane_center_source_y_m','final_actor_speed_mps'):
   action_scenario.CONFIG.pop(key,None)
  action_scenario.CONFIG.update(phases=validate_phases(spec['edit']['parameters']['phases']),action_program=PROGRAM,
      interpretation='host-compiled multi-stage original-vehicle action sequence; not reactive traffic',direction='registered lane intents selected by action sequence')
  try:action_scenario.generate(staging/'scenario',baseline)
  finally:action_scenario.CONFIG=original
 scenario=read(staging/'scenario/scenario.json')
 if not scenario['passed']:raise ValueError('Edit audit failed')
 if scenario['config']==read(Path(stable['release'])/'scenario/scenario.json')['config']:raise ValueError('New task must change the edit')
 edit=folder/'edit';edit.mkdir(exist_ok=True)
 for name in ('scenario.json','target_track.npz'):(edit/name).write_bytes((staging/'scenario'/name).read_bytes())
 spec['edit_receipts']={name:asset(edit/name) for name in ('scenario.json','target_track.npz')}
 if entry['id'] in m.get('edit_provenance',{}):spec['llm_edit_design']=m['edit_provenance'][entry['id']]
 write(taskfile,spec)
 from feasibility import check
 witness=Path(stable['metrics']['cutin551']['report']['path'])
 result=check(taskfile,witness);write(folder/'feasibility.json',result)
 return taskfile


def admit(m,taskfile,measurements,qualification):
 task=read(taskfile);folder=taskfile.parent;parent=read(m['policies']['global']);ms=measurements['global'];profile='v15_joint'
 champion=dict(parent,version=parent['version'],profile=profile,metrics=ms,reference_metrics=copy.deepcopy(ms),matrix=str(folder/'global'))
 # A failed target is kept as failed. It cannot form a healthy Pareto frontier.
 registry={} if not healthy(ms['cutin551']) else {parent['version']:dict(champion,pool_id=parent['version'],eligible=True,evaluated_repairs=0)}
 if registry:champion['pool_id']=parent['version']
 failure=dict(schema='fixed_v15_new_edit_failure_v2',classification=qualification['classification'],qualification=qualification,edit=task['edit'],measurements=measurements,diagnosis='Diagnose the inherited global v15 in this NEW edit. No v17 test or pre-existing better controller is required. Preserve original/legacy behavior and verify minimal causal repair.',physical_failure_claim=False)
 write(task['failure_reason'],failure)
 write(task['admission'],dict(qualified=True,scene=551,edit=task['id'],fresh_baseline=True,carried_policy_sha256=sha(Path(parent['code'])/'joint_safety.py'),champion=champion,registry=registry,qualification=qualification))
 write(folder/'edit_instruction.json',dict(edit=task['edit'],assets=task['edit_receipts']))
 write(OUT/'selected_task.json',dict(task=asset(taskfile),admission=asset(task['admission']),failure_reason=asset(task['failure_reason']),automatic_handoff=True))
 return taskfile


def candidate_prepare(taskfile,policyfile,out):
 out=Path(out);frozen={}
 for case in ('original551','cutin551','legacy551'):
  work=out/'cases'/case/'on'
  subprocess.run([str(PYTHON),'-B',str(LOCAL/'probe_worker.py'),'prepare','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case],cwd=ROOT,check=True)
  frozen.update(read(work/'input_pins.json'))
 write(out/'prepared.json',dict(task=asset(taskfile),policy=asset(policyfile),frozen_inputs=frozen))


def candidate_gpu(taskfile,policyfile,out):
 out=Path(out);verify_pins(read(out/'prepared.json')['frozen_inputs']);result={}
 for case in ('original551','cutin551','legacy551'):
  if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('GPU STOP present')
  work=out/'cases'/case/'on'
  subprocess.run([str(PYTHON),'-u','-B',str(LOCAL/'probe_worker.py'),'execute','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case],cwd=ROOT,check=True,pass_fds=(9,))
  result[case]=metrics(work,'A' if case=='original551' else 'B')
 write(out/'evaluation.json',dict(status='paired_new_edit_and_legacy_evaluated',metrics={k:result[k] for k in ('original551','cutin551')},legacy_metrics=dict(original551=result['original551'],cutin551=result['legacy551']),missing_cases=['original172','original552','original621'],global_patch_validated=False,task=asset(taskfile)))


def handoff(m,taskfile):
 sys.path.insert(0,str(AGENT))
 from recovery import RecoveryEngine as Engine
 from task_adapter import NewEditAdapter
 import convergence
 task=read(taskfile);config=read(OUT/'source_config.json');entry=dict(id=task['id'],scene=551,edit=task['id'],ready=True,task_file=str(taskfile),task_asset=asset(taskfile),adapter=dict(module=asset(LOCAL/'task_adapter.py'),factory='make_adapter'))
 if m.get('scene_action_cycle',{}).get('enabled'):
  from blind_config import configuration
  config['generalization']=configuration(m,taskfile);config['scene_cycle']={'different_scene_after_confirmation':True}
  suite=OUT/'action_validation/suite.json'
  config['public_training_audit']=dict(enabled=suite.exists(),suite=asset(suite) if suite.exists() else None)
 next_scenes=[dict(id='scene_'+str(scene)+'_action_discovery',scene=scene,edit='new_LLM_action_discovery',ready=False,unavailable_reason='Matching runtime/map/scorer and discovery adapter must be admitted') for scene in (172,552,621) if scene!=task['scene']] if m.get('scene_action_cycle',{}).get('enabled') else []
 config['task_queue']=dict(current_id=task['id'],tasks=[entry,*next_scenes]);config['candidate_registry']=[];config['presentation']['enabled']=False
 config['external_new_task_host_pins']=m['host_pins'];config.setdefault('acceptance',{})['lateral_diagnostics_only']=True;config['convergence']['allow_control_failure_baseline']=True;config['editable_methods']=['JointReference.rollout'];old=read(OUT/'source_state.json');ledger=read(OUT/'ledger.json')
 # Supplemental allocation is explicit; all previous API/token/round/GPU counters carry forward.
 config['budgets']['wall_seconds']=remaining(m)
 config['budgets']['max_gpu_attempts']+=m['explicit_budget_extension']['gpu_attempts']
 campaign='new_edit_'+task['id'];root=P26/'autonomous_patch_agent_v8/campaigns'/campaign
 adapter=NewEditAdapter(entry)
 if not (root/'state.json').exists():
  engine=Engine(root,config,adapter=adapter);state=engine.initialize(campaign)
  from migration import merge_design_usage
  usage_file=OUT/'scene_design/usage.json';design_usage=read(usage_file) if usage_file.exists() else {}
  state['usage']=merge_design_usage(old['usage'],design_usage)
  validation_usage=OUT/'action_validation/design/usage.json'
  if validation_usage.exists():state['usage']=merge_design_usage(state['usage'],read(validation_usage))
  delta=ledger['gpu_attempts']-m.get('discovery_gpu_carried',0)
  state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+delta;state['usage']['gpu_dispatches']=state['usage'].get('gpu_dispatches',0)+delta
  state['rounds_started']=old['rounds_started'];state['active_task_spec']=entry;state['prior_campaign_history']=old['history'];state['started_epoch']=time.time()
  state['budget_lineage']=dict(source=asset(OUT/'source_state.json'),discovery_ledger=asset(OUT/'ledger.json'),extension=m['explicit_budget_extension'],counters_reset=False)
  # The new baseline is measured anew; no old-task scores enter the new frontier.
  engine.save(state)
 else:engine=Engine(root)
 write(OUT/'handoff.json',dict(campaign=str(root),task=asset(taskfile),automatic=True,counters_reset=False))
 status('patch_agent_running',campaign=str(root))
 state=engine.run();status('patch_agent_'+state['status'],campaign=str(root),active_task=state.get('active_task_id'))


def run():
 from migration import acquire_previous,settle_pending
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'worker.lock').open('a') as own_lock:
  fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior_lock=acquire_previous(status,lambda:(OUT/'STOP').exists())
  try:
   if not (OUT/'mission.json').exists():settle_pending(status)
   options=read(OUT/'launch_options.json') if (OUT/'launch_options.json').exists() else {}
   m=initialize(options.get('add_wall_seconds'),options.get('add_gpu_attempts'))
   verify_pins(m['host_pins']);verify(m['global_patch'])
   if (OUT/'selected_task.json').exists():handoff(m,Path(read(OUT/'selected_task.json')['task']['path']));return
   from scene_designer import Designer,compile_entries
   from context_builder import build
   source=read(OUT/'source_state.json');config=read(OUT/'source_config.json')
   designer=Designer(OUT/'scene_design',config,source['usage'],wall_remaining=lambda:remaining(m))
   feedback=[]
   for batch in range(1,m['scenario_design']['max_batches']+1):
    if (OUT/'STOP').exists():status('stopped');return
    remaining(m)
    if read(OUT/'ledger.json')['gpu_attempts']>=m['explicit_budget_extension']['gpu_attempts']:
     sys.path.insert(0,str(AGENT));from core import BudgetExceeded
     raise BudgetExceeded('Discovery GPU budget exhausted before scene-design API request')
    context_file=OUT/'scene_design'/('batch_%03d'%batch)/'context.json'
    context=read(context_file) if context_file.exists() else build(m,feedback)
    status('llm_designing_scene_edits',batch=batch,global_default='v15')
    proposal=designer.propose(context,batch,m['scenario_design']['max_requests_per_batch'])
    entries=compile_entries(proposal,batch)
    if not entries:status('waiting_model_scene_design',reason=proposal['proposal']['note']);return
    origin=asset(OUT/'scene_design'/('batch_%03d'%batch)/'proposal.json')
    m.setdefault('edit_provenance',{})
    for entry in entries:
     existing=next((x for x in m['new_edits'] if x['id']==entry['id']),None)
     if existing is not None and existing!=entry:raise RuntimeError('Model-authored edit ID collision')
     if existing is None:m['new_edits'].append(entry)
     m['edit_provenance'][entry['id']]=origin
    write(OUT/'mission.json',m)
    for entry in entries:
     if (OUT/'STOP').exists():status('stopped');return
     status('auditing_llm_edit',batch=batch,edit=entry['id'])
     try:subprocess.run([str(PYTHON),'-u','-B',str(LOCAL/'flow.py'),'_generate','--edit',entry['id']],cwd=ROOT,check=True,env=native_environment())
     except subprocess.CalledProcessError:
      audit=OUT/'tasks'/entry['id']/'edit_work/scenario/scenario.json'
      if audit.exists() and read(audit).get('passed') is False:
       rejected=dict(edit=entry['id'],classification='invalid_edit_before_GPU',checks=read(audit)['checks'],motion=read(audit)['metrics'],phase_motion=read(audit).get('phase_motion',[]));feedback.append(rejected);write(audit.parent.parent.parent/'discovery_rejection.json',rejected);continue
      raise
     taskfile=OUT/'tasks'/entry['id']/'task.json';measurements={'global':{}}
     feasible=read(taskfile.parent/'feasibility.json')
     if not feasible['passed']:
      rejected=dict(edit=entry['id'],classification='offline_feasible_path_not_verified',minimum_gap=feasible['minimum_edited_actor_gap_m']);feedback.append(rejected);write(taskfile.parent/'discovery_rejection.json',rejected);continue
     parent=read(m['policies']['global']);original=parent['metrics']['original551']
     for k in ('report','summary','contract'):verify(original[k])
     measurements['global']['original551']=copy.deepcopy(original)
     status('probing_llm_edit_with_v15',batch=batch,edit=entry['id'],case='cutin551')
     baseline=probe(m,taskfile,m['policies']['global'],taskfile.parent/'global/cases/cutin551/on','cutin551');measurements['global']['cutin551']=baseline
     finding=assess_failure(baseline,original);write(taskfile.parent/'first_screen.json',finding)
     feedback.append(dict(edit=entry['id'],parameters={k:v for k,v in entry.items() if k!='id'},classification='measured_candidate' if finding['candidate'] else 'no_measured_failure',finding=finding,controls=baseline['controls'],hdscore=baseline['score']['hdscore']))
     if not finding['candidate']:continue
     status('repeating_v15_failure',edit=entry['id'])
     repeat=probe(m,taskfile,m['policies']['global'],taskfile.parent/'repeat/cases/cutin551/on','cutin551');same_world(baseline,repeat)
     q=qualify_fixed(baseline,repeat,original,feasible['passed']);write(taskfile.parent/'qualification.json',dict(**q,measurements=measurements,repeat=repeat))
     if q['qualified']:admit(m,taskfile,measurements,q);handoff(m,taskfile);return
    write(OUT/'scene_design'/('batch_%03d'%batch)/'measured_feedback.json',feedback)
   status('waiting_new_model_edit_batch',reason='Bounded model batches completed; no qualifying failure invented')
  finally:
   fcntl.flock(prior_lock,fcntl.LOCK_UN);prior_lock.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['start','resume','run','brief','doctor','stop','_generate','_prepare','_gpu']);p.add_argument('--add-wall-seconds',type=int);p.add_argument('--add-gpu-attempts',type=int);p.add_argument('--edit');p.add_argument('--task');p.add_argument('--policy');p.add_argument('--output');a=p.parse_args()
 if a.stage=='doctor':
  sys.path.insert(0,str(AGENT));from provider import HTTPProvider
  config=read(AGENT/'config.example.json');missing=HTTPProvider(config['api']).readiness()
  selected=read(P26/'global_patch.json');verify_pins(selected['policy']['code_pins'])
  print(json.dumps(dict(api_ready=not missing,missing_api_variables=missing,global_default=selected['version'],model_scene_design_enabled=True,admitted_scenes=[551],supported_edit_parameters=['phases'],supported_actions=['lane_change','decelerate','accelerate','cruise','stop_hold'],API_request_sent=False,GPU_started=False),indent=2));return
 if a.stage=='_generate':task_prepare(read(OUT/'mission.json'),next(x for x in read(OUT/'mission.json')['new_edits'] if x['id']==a.edit));return
 if a.stage=='_prepare':candidate_prepare(a.task,a.policy,a.output);return
 if a.stage=='_gpu':candidate_gpu(a.task,a.policy,a.output);return
 if a.stage=='brief':
  st=read(OUT/'status.json') if (OUT/'status.json').exists() else read(OUT/'startup.json') if (OUT/'startup.json').exists() else {'stage':'not_started'}
  print('Status: %s | Global: %s | Edit: %s | Case: %s'%(st['stage'],read(P26/'global_patch.json')['version'],st.get('edit','-'),st.get('case','-')))
  if st.get('missing_api_variables'):print('Missing API variables: '+', '.join(st['missing_api_variables']))
  if st.get('error'):print('Launch error: '+st['error'])
  if st['stage']=='not_started':print('No launch record; run llm_tasks.sh launch first')
  if (OUT/'owner.json').exists():
   sys.path.insert(0,str(DISCOVERY));from service import alive
   print('Coordinator alive: '+str(alive(read(OUT/'owner.json'))))
  if (OUT/'ledger.json').exists():
   ledger=read(OUT/'ledger.json');mission=read(OUT/'mission.json');left=mission['explicit_budget_extension']['wall_seconds']-(time.time()-ledger['started_epoch']) if ledger['started_epoch'] is not None else mission['explicit_budget_extension']['wall_seconds']
   print('Discovery GPU attempts: %s / %s | Remaining wall: %ss'%(ledger['gpu_attempts'],mission['explicit_budget_extension']['gpu_attempts'],max(0,round(left))))
  if (OUT/'scene_design/usage.json').exists():
   du=read(OUT/'scene_design/usage.json');print('Scene-design API calls: %s | Reported tokens: %s'%(du['api_requests'],du['reported_tokens']))
  if (OUT/'handoff.json').exists():
   campaign=Path(read(OUT/'handoff.json')['campaign']);state=read(campaign/'state.json');r=state.get('active_round') or {};u=state['usage']
   print('Agent: %s | Round: %s | Phase: %s'%(state['status'],r.get('id','-'),r.get('phase','-')))
   print('API calls: %s | Reported tokens: %s'%(u['api_requests'],u['reported_tokens']))
  return
 if a.stage=='stop':write(OUT/'STOP',dict(reason='operator stop'));return
 if a.stage in ('start','resume'):
  OUT.mkdir(parents=True,exist_ok=True)
  if any(v is not None and v<0 for v in (a.add_wall_seconds,a.add_gpu_attempts)):raise ValueError('Budget additions must be nonnegative')
  requested=dict(add_wall_seconds=a.add_wall_seconds,add_gpu_attempts=a.add_gpu_attempts)
  options=OUT/'launch_options.json'
  if not options.exists():write(options,requested)
  elif any(v is not None for v in requested.values()) and read(options)!=requested:raise ValueError('Existing allocation is immutable; resume without flags. Prior counters/timer are retained')
  if a.stage=='resume' and (OUT/'STOP').exists():(OUT/'STOP').unlink()
  sys.path.insert(0,str(DISCOVERY));from service import alive,identity
  with (OUT/'start.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owner=read(OUT/'owner.json') if (OUT/'owner.json').exists() else {}
   if alive(owner):print('already_running');return
   with (OUT/'worker.log').open('a') as log:proc=subprocess.Popen([str(PYTHON),'-u','-B',str(LOCAL/'flow.py'),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   write(OUT/'owner.json',dict(pid=proc.pid,identity=identity(proc.pid)));print('started',proc.pid,'log',OUT/'worker.log');return
 try:run()
 except Exception as e:
  sys.path.insert(0,str(AGENT))
  from core import Deferred,BudgetExceeded
  if isinstance(e,Deferred):status('waiting_api_scene_design',reason=str(e));return
  if isinstance(e,BudgetExceeded):status('budget_exhausted',reason=str(e),budget_reset=False);return
  import traceback
  status('blocked',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
