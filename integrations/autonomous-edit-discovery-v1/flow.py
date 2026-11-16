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
 return {str(p):sha(p) for folder in (LOCAL,DISCOVERY) for p in folder.rglob('*') if p.suffix in ('.py','.sh') and '__pycache__' not in p.parts}


def status(stage,**extra):write(OUT/'status.json',dict(stage=stage,updated_epoch=time.time(),**extra))


def initialize(extra_wall,extra_gpu):
 if (OUT/'mission.json').exists():
  m=read(OUT/'mission.json');verify_pins(m['host_pins']);return m
 source=read(SOURCE_CAMPAIGN/'state.json');pool=read(SOURCE_CAMPAIGN/'candidate_registry.json')
 eligible=[v for v in pool.values() if all(healthy(x) for x in v['metrics'].values())]
 frontier=max(eligible,key=lambda x:x['metrics']['cutin551']['score']['hdscore'])
 OUT.mkdir(parents=True,exist_ok=True)
 write(OUT/'source_state.json',source);write(OUT/'source_registry.json',pool);write(OUT/'source_config.json',read(SOURCE_CAMPAIGN/'config.json'))
 policies=dict(stable=source['champion'],frontier=frontier)
 for key,policy in policies.items():write(OUT/(key+'.json'),policy)
 m=dict(schema='new_edit_to_v7_patch_mission_v1',host_pins=frozen_host(),policies={k:str(OUT/(k+'.json')) for k in policies},
        explicit_budget_extension=dict(wall_seconds=extra_wall,gpu_attempts=extra_gpu,old_counters_preserved=True),
        source_state=asset(OUT/'source_state.json'),source_registry=asset(OUT/'source_registry.json'),new_edits=EDIT_GRID,
        generation_scope='new frozen trajectory edits on admitted551; other scenes remain gated by per-scene runtime/map/evaluator admission',
        baseline='inherited accepted controller, not disabling all previous accepted patches',automatic_handoff=True,global_patch_validated=False)
 write(OUT/'mission.json',m);write(OUT/'ledger.json',dict(started_epoch=None,gpu_attempts=0));return m


def remaining(m):
 ledger=read(OUT/'ledger.json')
 if ledger['started_epoch'] is None:ledger['started_epoch']=time.time();write(OUT/'ledger.json',ledger)
 value=m['explicit_budget_extension']['wall_seconds']-(time.time()-ledger['started_epoch'])
 if value<=0:raise RuntimeError('Supplemental mission wall budget exhausted; counters retained')
 return value


def probe(m,taskfile,policyfile,work,case):
 work=Path(work);arm='A' if case=='original551' else 'B'
 command=[str(PYTHON),'-u','-B',str(LOCAL/'probe_worker.py'),'prepare','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case]
 subprocess.run(command,cwd=ROOT,check=True)
 if (work/'completion.json').exists():return metrics(work,arm)
 if (work/arm).exists():raise RuntimeError('Partial episode retained; no automatic action replay: '+str(work))
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
 while True:
  remaining(m)
  if (OUT/'STOP').exists():raise RuntimeError('Operator stop requested')
  ledger=read(OUT/'ledger.json')
  if ledger['gpu_attempts']>=m['explicit_budget_extension']['gpu_attempts']:raise RuntimeError('Supplemental discovery GPU budget exhausted')
  attempt=ledger['gpu_attempts'];log=work/('gpu_attempt_%03d.log'%attempt)
  args=['bash',str(LOCAL/'run.sh'),'_probe','--task',str(taskfile),'--policy',str(policyfile),'--work',str(work),'--case',case]
  with log.open('x') as stream:
   proc=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
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
 spec=dict(schema='new_edit_task_v1',id='new_'+entry['id'],scene=551,edit=dict(id=entry['id'],scene=551,program='registered_native_cutin_duration_variants',parameters={k:v for k,v in entry.items() if k!='id'}),
           baseline_root=str(baseline),source_contracts={c:stable['metrics'][c]['contract']['path'] for c in stable['metrics']},
           edit_assets=str(folder/'edit'),admission=str(folder/'admission.json'),failure_reason=str(folder/'failure_reason.json'),
           source_state=str(OUT/'source_state.json'),source_registry=str(OUT/'source_registry.json'),host_pins=m['host_pins'])
 staging=folder/'edit_work'
 if not staging.exists():prepare_episode(spec['edit'],stable,staging,baseline=baseline)
 scenario=read(staging/'scenario/scenario.json')
 if not scenario['passed']:raise ValueError('Edit audit failed')
 if scenario['config']==read(Path(stable['release'])/'scenario/scenario.json')['config']:raise ValueError('New task must change the edit')
 edit=folder/'edit';edit.mkdir(exist_ok=True)
 for name in ('scenario.json','target_track.npz'):(edit/name).write_bytes((staging/'scenario'/name).read_bytes())
 spec['edit_receipts']={name:asset(edit/name) for name in ('scenario.json','target_track.npz')}
 write(taskfile,spec);return taskfile


def admit(m,taskfile,measurements,qualification):
 task=read(taskfile);folder=taskfile.parent;registry={}
 stable=read(m['policies']['stable']);frontier=read(m['policies']['frontier'])
 for key,parent in [('stable',stable),('frontier',frontier)]:
  ms=measurements[key];pool_id='v17' if key=='stable' else 'v15'
  registry[pool_id]=dict(parent,version=pool_id,pool_id=pool_id,metrics=ms,reference_metrics=measurements['stable'],matrix=str(folder/key),profile=parent.get('profile','v17_spatial' if key=='stable' else 'v15_joint'),eligible=True,evaluated_repairs=0)
 champion=copy.deepcopy(registry['v17']);champion['reference_metrics']=measurements['stable']
 failure=dict(schema='discovered_new_edit_failure_v1',classification=qualification['classification'],qualification=qualification,edit=task['edit'],measurements=measurements,diagnosis='Inherited controller has a repeatable lower executed native score than an independently executed alternative in this NEW edited world; the LLM must causally diagnose the mechanism before proposing a minimal patch.',truth_used_online=False,physical_failure_claim=False)
 write(task['failure_reason'],failure)
 write(task['admission'],dict(qualified=True,scene=551,edit=task['id'],fresh_baseline=True,carried_policy_sha256=sha(Path(stable['code'])/'joint_safety.py'),champion=champion,registry=registry,qualification=qualification))
 write(folder/'edit_instruction.json',dict(edit=task['edit'],assets=task['edit_receipts'],source='host-registered new edit; not the old canonical edit'))
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
 from engine import Engine
 from task_adapter import NewEditAdapter
 import convergence
 task=read(taskfile);config=read(OUT/'source_config.json');entry=dict(id=task['id'],scene=551,edit=task['id'],ready=True,task_file=str(taskfile),task_asset=asset(taskfile),adapter=dict(module=asset(LOCAL/'task_adapter.py'),factory='make_adapter'))
 config['task_queue']=dict(current_id=task['id'],tasks=[entry]);config['candidate_registry']=[];config['presentation']['enabled']=False
 config['external_new_task_host_pins']=m['host_pins'];old=read(OUT/'source_state.json');ledger=read(OUT/'ledger.json')
 # Supplemental allocation is explicit; all previous API/token/round/GPU counters carry forward.
 config['budgets']['wall_seconds']=remaining(m)
 config['budgets']['max_gpu_attempts']+=m['explicit_budget_extension']['gpu_attempts']
 campaign='new_edit_'+task['id'];root=P26/'autonomous_patch_agent_v7/campaigns'/campaign
 adapter=NewEditAdapter(entry)
 if not (root/'state.json').exists():
  engine=Engine(root,config,adapter=adapter);state=engine.initialize(campaign)
  state['usage']=copy.deepcopy(old['usage']);state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+ledger['gpu_attempts'];state['usage']['gpu_dispatches']=state['usage'].get('gpu_dispatches',0)+ledger['gpu_attempts']
  state['rounds_started']=old['rounds_started'];state['active_task_spec']=entry;state['prior_campaign_history']=old['history'];state['started_epoch']=time.time()
  state['budget_lineage']=dict(source=asset(OUT/'source_state.json'),discovery_ledger=asset(OUT/'ledger.json'),extension=m['explicit_budget_extension'],counters_reset=False)
  # The new baseline is measured anew; no old-task scores enter the new frontier.
  engine.save(state)
 else:engine=Engine(root,adapter=adapter)
 write(OUT/'handoff.json',dict(campaign=str(root),task=asset(taskfile),automatic=True,counters_reset=False))
 status('patch_agent_running',campaign=str(root))
 state=engine.run();status('patch_agent_'+state['status'],campaign=str(root),active_task=state.get('active_task_id'))


def run():
 m=read(OUT/'mission.json');verify_pins(m['host_pins'])
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'selected_task.json').exists():handoff(m,Path(read(OUT/'selected_task.json')['task']['path']));return
  for entry in m['new_edits']:
   if (OUT/'STOP').exists():status('stopped');return
   status('generating_new_edit',edit=entry['id'])
   try:subprocess.run([str(PYTHON),'-u','-B',str(LOCAL/'flow.py'),'_generate','--edit',entry['id']],cwd=ROOT,check=True)
   except subprocess.CalledProcessError:
    audit=OUT/'tasks'/entry['id']/'edit_work/scenario/scenario.json'
    if audit.exists() and read(audit).get('passed') is False:
     write(audit.parent.parent.parent/'discovery_rejection.json',dict(reason='illegal_edit_filtered_before_GPU',audit=asset(audit)));continue
    raise
   taskfile=OUT/'tasks'/entry['id']/'task.json';measurements={}
   for key in ('stable','frontier'):
    measurements[key]={}
    for case in ('original551','cutin551'):
     work=taskfile.parent/key/'cases'/case/'on';status('probing_inherited_controller' if key=='stable' else 'checking_feasible_alternative',edit=entry['id'],policy=key,case=case)
     measurements[key][case]=probe(m,taskfile,m['policies'][key],work,case)
   repeat=probe(m,taskfile,m['policies']['stable'],taskfile.parent/'repeat/cases/cutin551/on','cutin551')
   from common_flow import same_world
   baseline=measurements['stable']['cutin551'];alternative=measurements['frontier']['cutin551']
   same_world(baseline,repeat);same_world(baseline,alternative)
   q=challenge(baseline,repeat,alternative,measurements['stable']['original551'])
   write(taskfile.parent/'qualification.json',dict(**q,measurements=measurements,repeat=repeat))
   if q['qualified']:
    admit(m,taskfile,measurements,q);handoff(m,taskfile);return
  status('waiting_new_edit_family',reason='No registered generated edit met reproducible-gap and feasibility criteria; no fabricated failure')


def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['start','resume','run','brief','stop','_generate','_prepare','_gpu']);p.add_argument('--extra-wall-seconds',type=int);p.add_argument('--extra-gpu-attempts',type=int);p.add_argument('--edit');p.add_argument('--task');p.add_argument('--policy');p.add_argument('--output');a=p.parse_args()
 if a.stage=='_generate':task_prepare(read(OUT/'mission.json'),next(x for x in EDIT_GRID if x['id']==a.edit));return
 if a.stage=='_prepare':candidate_prepare(a.task,a.policy,a.output);return
 if a.stage=='_gpu':candidate_gpu(a.task,a.policy,a.output);return
 if a.stage=='brief':
  print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started');return
 if a.stage=='stop':write(OUT/'STOP',dict(reason='operator stop'));return
 if a.stage in ('start','resume'):
  if not (OUT/'mission.json').exists() and (not a.extra_wall_seconds or not a.extra_gpu_attempts):raise ValueError('First start requires explicit supplemental wall/GPU allocation; old budgets are never reset')
  initialize(a.extra_wall_seconds,a.extra_gpu_attempts)
  if a.stage=='resume' and (OUT/'STOP').exists():(OUT/'STOP').unlink()
  sys.path.insert(0,str(DISCOVERY));from service import alive,identity
  with (OUT/'start.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owner=read(OUT/'owner.json') if (OUT/'owner.json').exists() else {}
   if alive(owner):print('already_running');return
   with (OUT/'worker.log').open('a') as log:proc=subprocess.Popen([str(PYTHON),'-u','-B',str(LOCAL/'flow.py'),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   write(OUT/'owner.json',dict(pid=proc.pid,identity=identity(proc.pid)));print('started',proc.pid,'log',OUT/'worker.log');return
 try:run()
 except BaseException as e:
  import traceback
  status('blocked',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
