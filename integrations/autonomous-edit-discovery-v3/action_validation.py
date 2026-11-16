"""Frozen same-scene validation actions, actual paired jobs, and default publication."""
import copy,fcntl,sys,time
from common_flow import *
sys.path.insert(0,str(AGENT))
from core import AgentError,Deferred,BudgetExceeded,digest,remaining
from scene_designer import Designer,compile_entries,signature
from action_program import validate_phases
from migration import merge_design_usage
import jobs

def family(phases):
 lane='original_lane';events=[]
 for p in validate_phases(phases):
  new_lane=lane if p['lane']=='keep' else p['lane']
  if p['action']!='cruise':events.append((p['action'],new_lane if new_lane!=lane else 'keep'))
  lane=new_lane
 return digest(events)

def reserve(m,taskfile):
 suitefile=OUT/'action_validation/suite.json'
 if suitefile.exists():
  suite=read(suitefile)
  if suite['training_task']!=asset(taskfile):raise AgentError('Validation split belongs to another training task')
  for case in suite['cases']:verify(case['task'])
  return suite
 import flow
 from context_builder import build
 train=read(taskfile);train_phases=train['edit']['parameters']['phases'];train_sig=signature(dict(phases=train_phases));train_family=family(train_phases)
 base=read(OUT/'source_state.json')['usage'];config=read(OUT/'source_config.json')
 if (OUT/'scene_design/usage.json').exists():base=merge_design_usage(base,read(OUT/'scene_design/usage.json'))
 designer=Designer(OUT/'action_validation/design',config,base,wall_remaining=lambda:flow.remaining(m))
 cases=[];families=set();seen={train_sig};feedback=[]
 for batch in range(1,4):
  folder=OUT/'action_validation/design'/('batch_%03d'%batch);contextfile=folder/'context.json'
  context=read(contextfile) if contextfile.exists() else build(m,feedback)
  if not contextfile.exists():
   context['known_parameter_signatures']=sorted(set(context['known_parameter_signatures'])|seen)
   context['reserved_validation_only']=dict(training_family_excluded=train_family,training_fingerprint_excluded=train_sig,goal='Propose mechanisms different from training. These actions are frozen BEFORE patch tuning. No validation scores enter training.')
  proposal=designer.propose(context,batch);entries=compile_entries(proposal,100+batch)
  for entry in entries:
   fingerprint=signature(dict(phases=entry['phases']));mechanism=family(entry['phases'])
   if fingerprint in seen or mechanism==train_family or mechanism in families:
    feedback.append(dict(edit=entry['id'],reason='Repeated or training action mechanism'));continue
   seen.add(fingerprint);original=flow.OUT;flow.OUT=OUT/'action_validation/assets'
   try:
    task=flow.task_prepare(m,entry)
    if not read(task.parent/'feasibility.json')['passed']:feedback.append(dict(edit=entry['id'],reason='No offline feasible-path witness'));continue
   except ValueError:
    audit=flow.OUT/'tasks'/entry['id']/'edit_work/scenario/scenario.json'
    if not audit.exists():raise
    feedback.append(dict(edit=entry['id'],reason='Invalid validation edit',checks=read(audit)['checks'],phase_motion=read(audit).get('phase_motion',[])));continue
   finally:flow.OUT=original
   cases.append(dict(id='held_action_'+fingerprint[:16],scene=train['scene'],kind='action',action_fingerprint=fingerprint,action_family=mechanism,withheld_from_training=True,ready=True,baseline=None,task=asset(task)))
   families.add(mechanism)
   if len(cases)>=2:break
  if len(cases)>=2:break
 if len(cases)<2:raise Deferred('Waiting for two legal, disjoint validation action mechanisms; no confirmation fabricated')
 cases.insert(0,dict(id='original_'+str(train['scene']),scene=train['scene'],kind='original',ready=True,baseline=None,task=asset(taskfile)))
 suite=dict(schema='frozen_scene_action_validation_v1',training_task=asset(taskfile),training_action_fingerprints=[train_sig],training_action_families=[train_family],cases=cases,baseline_policy=asset(m['policies']['global']),global_parent=asset(P26/'global_patch.json'),frozen_epoch=time.time(),scores_hidden_from_patch_training=True)
 write(suitefile,suite);return suite

def configuration(m,taskfile):
 suite=reserve(m,taskfile);parent=read(suite['baseline_policy']['path'])
 return dict(required=True,mode='same_scene_actions',training_scene=read(taskfile)['scene'],training_base_pins=parent['code_pins'],training_action_fingerprints=suite['training_action_fingerprints'],training_action_families=suite['training_action_families'],suite=asset(OUT/'action_validation/suite.json'),baseline_policy=suite['baseline_policy'],global_parent=suite['global_parent'],cases=suite['cases'],score_tolerance=1e-9)

def prepare_case(case,policy,folder):
 import probe_worker
 verify(case['task']);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 write(folder/'case.json',case);write(folder/'policy.json',policy)
 probe_worker.stage(case['task']['path'],folder/'policy.json',folder/'episode','original551' if case['kind']=='original' else 'cutin551')
 return folder

def prepare_baselines(state,config,root):
 root=Path(root);p=config['generalization'];verify(p['suite']);verify(p['baseline_policy']);policy=read(p['baseline_policy']['path']);verify_pins(policy['code_pins'])
 state.setdefault('action_validation_baselines',{})
 resolved=[]
 for case in p['cases']:
  folder=root/'action_validation_baselines'/case['id'];proof=folder/'baseline.json'
  if proof.exists():
   receipt=jobs.inspect(folder/'job')
   if receipt['status']=='running':state['status']='running_action_validation_baselines';return None
   if receipt['status']!='succeeded':state['status']='action_validation_baseline_infrastructure_failure';return False
   record=read(proof)
   if record['code_pins']!=policy['code_pins'] or read(record['comparison_contract']['path']).get('action_fingerprint')!=case.get('action_fingerprint'):raise AgentError('Cached baseline belongs to a different policy/action')
   resolved.append(dict(case,baseline=asset(proof)));continue
  prepare_case(case,policy,folder);inputs=read(folder/'episode/input_pins.json');job=folder/'job'
  if case['id'] not in state['action_validation_baselines']:
   if state['usage'].get('gpu_attempts',0)>=config['budgets']['max_gpu_attempts']:raise BudgetExceeded('Action validation shares GPU budget')
   state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;state['usage']['gpu_dispatches']+=1;state['action_validation_baselines'][case['id']]=dict(reserved=True);write(root/'state.json',state)
  result=jobs.launch(job,['bash',str(LOCAL/'run.sh'),'_gate','--folder',str(folder)],ROOT,min(config['budgets']['gpu_stage_seconds'],remaining(state,config)),{**inputs,**policy['code_pins']},[str(proof)],root/'STOP',)
  if result['status']=='running':state['status']='running_action_validation_baselines';return None
  if result['status']!='succeeded':state['status']='action_validation_baseline_infrastructure_failure';return False
  resolved.append(dict(case,baseline=asset(proof)))
 return resolved

def candidate_stage(case,candidate,folder,config):
 folder=prepare_case(case,candidate,folder);write(folder/'baseline_binding.json',case['baseline'])
 baseline=read(case['baseline']['path']);verify(baseline['comparison_contract'])
 admission=dict(passed=True,case_id=case['id'],candidate_pins=candidate['code_pins'],baseline=case['baseline']);write(folder/'admission.json',admission)
 return dict(admission=asset(folder/'admission.json'),inputs={**read(folder/'episode/input_pins.json'),**candidate['code_pins'],str(folder/'case.json'):sha(folder/'case.json'),str(folder/'baseline_binding.json'):sha(folder/'baseline_binding.json')},command=['bash',str(LOCAL/'run.sh'),'_gate','--folder',str(folder)],outputs=[str(folder/'proof.json')],timeout=config['budgets']['gpu_stage_seconds'],cwd=ROOT)

def execute(folder):
 import os
 os.fstat(9)
 import probe_worker
 from governance import SHARED
 folder=Path(folder);case=read(folder/'case.json');policy=read(folder/'policy.json');work=folder/'episode'
 stamp=folder/'execution_start.json'
 if not stamp.exists():write(stamp,dict(started_epoch=time.time()))
 metric=probe_worker.run(case['task']['path'],folder/'policy.json',work,'original551' if case['kind']=='original' else 'cutin551')
 raw=read(metric['report']['path']);native=read(metric['contract']['path']);plan=read(work/'plan.json')
 binding=folder/'baseline_binding.json'
 if binding.exists():
  base=read(binding);verify(base);baseline=read(base['path']);shared=baseline['comparison_contract'];verify(shared)
  baseline_native=read(read(shared['path'])['native_contract']['path'])
  if any(native[k]!=baseline_native[k] for k in SHARED):raise AgentError('Action validation exogenous worlds differ')
 else:
  scorer=asset(ROOT/'code/HUGSIM/sim/utils/score_calculator.py');write(folder/'score_config.json',dict(score_keys=['hdscore','pdms','rc'],score_kind='executed_path',native_code=scorer))
  common=dict(scene=case['scene'],scene_manifest=asset(work/'scene_manifest.json'),scenario=asset(work/'scenario/scenario.json'),initial_context=plan['initial_context'],source_end_frame=plan['source_end_frame'],seed=plan['seed'],model_weights=plan['assets']['uniad_weights'],static_map=asset(work/'map/map.json'),scorer=scorer,score_config=asset(folder/'score_config.json'),physics=asset(work/'scene_manifest.json'),action_fingerprint=case.get('action_fingerprint'),native_contract=metric['contract'])
  write(folder/'comparison.json',common);shared=asset(folder/'comparison.json')
 contract=dict(scene=case['scene'],comparison_contract=shared,code_pins=policy['code_pins'],fresh_history=True,score_kind='executed_path',history_run_id=str(work.resolve()),controls=plan['controls'],source_contract=metric['contract'])
 write(folder/'contract.json',contract);write(folder/'plan.json',dict(comparison_contract=shared,source_plan=asset(work/'plan.json')))
 report=dict(raw,validation_contract=asset(folder/'contract.json'),experiment_plan=asset(folder/'plan.json'),started_epoch=read(stamp)['started_epoch'],source_report=metric['report'])
 write(folder/'report.json',report);episode=dict(report=asset(folder/'report.json'),summary=metric['summary'],contract=asset(folder/'contract.json'))
 if binding.exists():write(folder/'proof.json',dict(case_id=case['id'],candidate_pins=policy['code_pins'],baseline=read(binding),comparison_contract=shared,candidate=episode))
 else:write(folder/'baseline.json',dict(code_pins=policy['code_pins'],comparison_contract=shared,episode=episode))

def publish(state,config,root):
 if config['generalization'].get('mode')!='same_scene_actions':return
 verify(state['accepted_patch']);package=read(state['accepted_patch']['path'])
 if state.get('simulation_only') or state.get('final_patch_status')!='confirmed':raise AgentError('Only a real confirmed gate can publish')
 if package.get('confirmed') is not True or package.get('validation_scope')!='same_scene_cross_action':raise AgentError('Missing actual cross-action confirmation')
 from generalization import policy as verify_policy,verify_case
 verify_policy(config);verify(package['request']);request=read(package['request']['path'])
 if request.get('mode')!='same_scene_actions' or request['candidate']['code_pins']!=state['champion']['code_pins'] or len(package['cases'])!=len(request['cases']):raise AgentError('Confirmed package/controller mismatch')
 expected={c['id']:c for c in config['generalization']['cases']}
 for c,result in zip(request['cases'],package['cases']):
  frozen=dict(c,baseline=None)
  if frozen!=expected.get(c['id']) or not verify_case(result['proof'],state['champion'],c,request['requested_epoch'],request['score_tolerance'])['passed']:raise AgentError('Action proof failed revalidation before publication')
 pointer=P26/'global_patch.json';root=Path(root);folder=root/'global_publication';folder.mkdir(exist_ok=True)
 with (P26/'global_patch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  current=read(pointer)
  if current.get('accepted_patch')==state['accepted_patch']:return
  verify(config['generalization']['global_parent']);write(folder/'previous_global_patch.json',current)
  policy=copy.deepcopy(state['champion']);policy['profile']='v15_joint'
  write(pointer,dict(schema='confirmed_scene_action_global_default_v1',version=policy['version'],policy=policy,accepted_patch=state['accepted_patch'],default_for_new_tasks=True,confirmed_default=True,user_confirmed_default=False,registered_generalization_validated=True,cross_scene_generalization_validated=False,validation_scope='same_scene_cross_action',previous_global=asset(folder/'previous_global_patch.json'),selection_authority='User-authorized scene/action/convergence workflow'))
  write(folder/'publication.json',dict(published=True,global_patch=asset(pointer),accepted_patch=state['accepted_patch'],budget_reset=False))
