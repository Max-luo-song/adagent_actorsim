"""Mandatory host-owned finalization gate; local champions remain provisional."""

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

import copy,time
pass  # Path is provided by the portable boundary above
from core import AgentError,BudgetExceeded,asset,digest,finite,read,remaining,verify,verify_pins,write,utc
import jobs
SCORE_KEYS=('hdscore','pdms','rc')

class ExecutedControlFailure(AgentError):
 def __init__(self,facts):self.facts=facts;super().__init__('Verified executed controller termination')

def policy(config):
 p=config.get('generalization',{})
 if p.get('mode')=='counterexample_cross_scene':
  from validation_lifecycle import policy as lifecycle_policy
  return lifecycle_policy(config)
 if p.get('mode')=='blind_cross_scene':
  from blind_validation import policy as private_policy
  return private_policy(config)
 if p.get('required') is not True:raise AgentError('Generalization is mandatory for real finalization')
 cases=p.get('cases',[]);ids=[x['id'] for x in cases]
 if not cases or len(ids)!=len(set(ids)):raise AgentError('Register unique generalization cases')
 mode=p.get('mode','cross_scene')
 if mode=='same_scene_actions':
  scenes={x['scene'] for x in cases};actions=[x for x in cases if x.get('kind')=='action']
  if scenes!={p.get('training_scene')} or not any(x.get('kind')=='original' for x in cases):raise AgentError('Same-scene gate needs original regression and matched action scene')
  if len(actions)<2 or len({x.get('action_family') for x in actions})<2:raise AgentError('Reserve at least two distinct action mechanisms')
  if len({x.get('action_fingerprint') for x in actions})!=len(actions):raise AgentError('Duplicate held-out action')
  if any(not x.get('withheld_from_training') or not x.get('action_fingerprint') or x['action_fingerprint'] in p.get('training_action_fingerprints',[]) or x.get('action_family') in p.get('training_action_families',[]) for x in actions):raise AgentError('Validation actions must be disjoint from training')
  verify(p['suite'])
  if read(p['suite']['path'])['cases']!=cases:raise AgentError('Frozen action suite changed')
 elif mode=='cross_scene':
  if len({x['scene'] for x in cases})<(1 if p.get('_host_single_sealed_scene') else 2):raise AgentError('Register distinct held-out scenes')
 else:raise AgentError('Unknown generalization mode')
 if not finite(p.get('score_tolerance')) or not 0<=p['score_tolerance']<=1e-9:raise AgentError('Only numerical score tolerance is allowed')
 return p

def comparison(a):
 verify(a);c=read(a['path'])
 keys={'scene','scene_manifest','scenario','initial_context','source_end_frame','seed','model_weights','static_map','scorer','score_config','physics'}
 if not keys<=set(c):raise AgentError('Incomplete exogenous comparison contract')
 for k in ['scene_manifest','scenario','model_weights','static_map','scorer','score_config','physics']:verify(c[k])
 if c.get('native_contract'):verify(c['native_contract'])
 return c

def measured(side,pins,shared,scene,not_before=None):
 verify_pins(pins)
 for k in ('report','summary','contract'):verify(side[k])
 r=read(side['report']['path']);s=read(side['summary']['path']);c=read(side['contract']['path'])
 if c.get('scene')!=scene or c.get('comparison_contract')!=shared or c.get('code_pins')!=pins:raise AgentError('Executed scene/controller/comparison mismatch')
 if r.get('validation_contract')!=side['contract'] or not r.get('real_closed_loop') or not r.get('model_weights_unchanged'):raise AgentError('Actual executed report is not bound to frozen contract')
 if r.get('source_report'):
  verify(r['source_report']);source=read(r['source_report']['path']);verify(c['source_contract'])
  if source.get('validation_contract')!=c['source_contract'] or any(source.get(k)!=r.get(k) for k in ('status','trace','observations','real_closed_loop','model_weights_unchanged')):raise AgentError('Derived gate report differs from actual execution')
 plan=r.get('experiment_plan');verify(plan)
 if read(plan['path']).get('comparison_contract')!=shared:raise AgentError('Plan does not bind common experiment inputs')
 if not c.get('fresh_history') or c.get('score_kind')!='executed_path' or not isinstance(c.get('history_run_id'),str) or not c['history_run_id']:raise AgentError('Fresh independent history is required')
 if not_before is not None and (not finite(r.get('started_epoch')) or r['started_epoch']<not_before):raise AgentError('Candidate must be newly executed after request')
 count=c.get('controls');trace=r.get('trace',[]);obs=r.get('observations',[])
 if type(count) is not int or count<25:raise AgentError('Invalid declared native horizon')
 if r.get('status')!='completed' or len(trace)!=count or len(obs)!=count+1:
  reason=r.get('termination_reason') or (r.get('road_safety_termination') or {}).get('reason')
  if r.get('status')=='terminated' and reason in ('no_jointly_verified_action','protective_stop_unresolved','physical_collision','authorized_static_map_departure') and 0<len(trace)<count and len(obs)==len(trace)+1 and not r.get('error') and not s.get('error_stage'):
   raise ExecutedControlFailure(dict(termination_reason=reason,controls=len(trace),declared_controls=count,history_run_id=c['history_run_id'],scores=None,complete_native_horizon=False))
  raise AgentError('Incomplete executed episode or native horizon')
 values=s['native_scores']['scores']['executed_path'];scores={k:values[k]['value'] for k in SCORE_KEYS}
 if any(not finite(v) or not 0<=v<=1 for v in scores.values()):raise AgentError('Invalid native scores')
 safety=bool(any(t.get('swept_collision',False) for t in trace) or s['anomalies']['environment_collision'] or s['stop_recovery_unresolved'] or any(o['environment'].get('authorized_static_map_departure',False) for o in obs))
 return dict(scores=scores,safety_failure=safety,controls=count,history_run_id=c['history_run_id'])

def verify_case(proof,candidate,case,requested_epoch,tolerance):
 verify(proof);p=read(proof['path']);verify_pins(candidate['code_pins']);base=case['baseline'];verify(base)
 registered=read(base['path'])
 if p.get('case_id')!=case['id'] or p.get('candidate_pins')!=candidate['code_pins'] or p.get('baseline')!=base:raise AgentError('Proof is not bound to registered candidate/baseline')
 if p['comparison_contract']!=registered['comparison_contract']:raise AgentError('Changed comparison contract')
 common=comparison(p['comparison_contract'])
 if common['scene']!=case['scene']:raise AgentError('Comparison belongs to different scene')
 if case.get('kind')=='action' and common.get('action_fingerprint')!=case['action_fingerprint']:raise AgentError('Proof belongs to another action')
 b=measured(registered['episode'],registered['code_pins'],p['comparison_contract'],case['scene'])
 try:n=measured(p['candidate'],candidate['code_pins'],p['comparison_contract'],case['scene'],requested_epoch)
 except ExecutedControlFailure as failure:
  if failure.facts['history_run_id']==b['history_run_id'] or failure.facts['declared_controls']!=b['controls']:raise AgentError('Control failure history/horizon mismatch')
  return dict(case_id=case['id'],scene=case['scene'],passed=False,reasons=['executed_control_failure'],score_deltas=None,baseline=b,candidate=failure.facts,proof=proof)
 if n['history_run_id']==b['history_run_id']:raise AgentError('Baseline/candidate cannot share perception history')
 if n['controls']!=b['controls']:raise AgentError('Episode horizons differ')
 deltas={k:n['scores'][k]-b['scores'][k] for k in SCORE_KEYS}
 reasons=[k+'_regression' for k,v in deltas.items() if v < -tolerance]
 if n['safety_failure']:reasons.append('physical_or_unresolved_stop_failure')
 return dict(case_id=case['id'],scene=case['scene'],passed=not reasons,reasons=reasons,score_deltas=deltas,baseline=b,candidate=n,proof=proof)

def request(state,config,root,cases_override=None):
 p=copy.deepcopy(policy(config));p['cases']=cases_override if cases_override is not None else p['cases'];root=Path(root);candidate=state['champion'];verify_pins(candidate['code_pins'])
 if p.get('mode')=='same_scene_actions':
  if not state.get('task_convergence',{}).get(state['active_task_id'],{}).get('converged'):raise AgentError('Action validation requires measured convergence first')
  if candidate['code_pins']==p['training_base_pins']:raise AgentError('No newly improved patch to confirm')
  for case in p['cases']:
   if not case.get('baseline'):raise AgentError('Real matched parent baseline missing')
   verify(case['baseline'])
   if read(case['baseline']['path'])['code_pins']!=p['training_base_pins']:raise AgentError('Validation baseline is not the inherited global controller')
 task=state['active_task_id'];gid=task+'_'+digest(candidate['code_pins'])[:16];folder=root/'generalization'/gid
 path=folder/'request.json'
 if path.exists():
  req=read(path)
  if req['candidate']['code_pins']!=candidate['code_pins'] or req['cases']!=p['cases']:raise AgentError('Frozen generalization request changed')
 else:
  req=dict(schema='patch_generalization_request_v1',id=gid,task_id=task,candidate=copy.deepcopy(candidate),cases=copy.deepcopy(p['cases']),requested_epoch=time.time(),time=utc(),score_tolerance=p['score_tolerance'],formal_certification=False,mode=p.get('mode','cross_scene'),validation_scope='same_scene_cross_action' if p.get('mode')=='same_scene_actions' else 'cross_scene')
  write(path,req)
 state['generalization']=dict(id=gid,request=asset(path),case_index=0,results=[],status='pending',current_job=None)
 state.update(status='generalization_pending',final_patch_status='pending_generalization',generalization_pending=True)
 return req

def finalize(state,req,root):
 g=state['generalization'];results=g['results']
 if len(results)!=len(req['cases']) or {r['case_id'] for r in results}!={x['id'] for x in req['cases']} or any(r.get('passed') is not True for r in results):raise AgentError('All registered scenes must pass before finalization')
 if state.get('simulation_only'):raise AgentError('Simulated tests cannot confirm a real patch')
 verify(g['request']);verify_pins(req['candidate']['code_pins'])
 # Revalidate all artifacts and scores immediately before the atomic pointer write.
 checked=[verify_case(r['proof'],req['candidate'],c,req['requested_epoch'],req['score_tolerance']) for c,r in zip(req['cases'],results)]
 if any(not x['passed'] for x in checked):raise AgentError('Generalization evidence regressed')
 root=Path(root);folder=root/'generalization'/req['id'];package=dict(schema='confirmed_patch_v1',candidate=req['candidate'],request=g['request'],cases=checked,time=utc(),confirmed=True,registered_generalization_validated=True,formal_certification=False,scope='Only the explicitly registered scenes and frozen inputs; not universal safety',production_driver_changed=False,validation_scope=req.get('validation_scope','cross_scene'),cross_scene_generalization_validated=req.get('mode','cross_scene')=='cross_scene',global_default_authorized=req.get('mode')=='same_scene_actions')
 write(folder/'accepted_patch.json',package);write(root/'accepted_patch.json',dict(package=asset(folder/'accepted_patch.json'),candidate=req['candidate'],confirmed=True))
 state.update(final_patch_status='confirmed',accepted_patch=asset(folder/'accepted_patch.json'),generalization_pending=False,registered_generalization_validated=True)
 g.update(status='passed',results=checked,current_job=None)

def step(state,config,adapter,root):
 if config.get('generalization',{}).get('mode')=='counterexample_cross_scene':
  from validation_lifecycle import step as lifecycle_step
  return lifecycle_step(state,config,adapter,root)
 if config.get('generalization',{}).get('mode')=='blind_cross_scene':
  from blind_validation import step as private_step
  return private_step(state,config,adapter,root)
 # Legacy CPU fixtures can exercise rotation, but never create accepted_patch.json.
 if state.get('simulation_only') and config.get('generalization_fixture_bypass') is True:return True
 policy(config);root=Path(root)
 if config['generalization'].get('mode')=='same_scene_actions':
  if not state.get('task_convergence',{}).get(state['active_task_id'],{}).get('converged'):raise AgentError('Action validation requires convergence before any GPU dispatch')
  if state['champion']['code_pins']==config['generalization']['training_base_pins']:raise AgentError('No newly improved patch to confirm')
 if not state.get('generalization'):
  resolved=None
  if config['generalization'].get('mode')=='same_scene_actions':
   preparer=getattr(adapter,'prepare_action_generalization',None)
   if preparer is None:state['status']='waiting_action_validation_adapter';return False
   resolved=preparer(state,config,root)
   if resolved is None:return None
   if resolved is False:return False
  request(state,config,root,resolved)
 elif state['generalization'].get('status')=='passed':
  previous=read(state['generalization']['request']['path'])
  if previous['task_id']!=state['active_task_id'] or previous['candidate']['code_pins']!=state['champion']['code_pins']:request(state,config,root)
 g=state['generalization'];verify(g['request']);req=read(g['request']['path'])
 if req['candidate']['code_pins']!=state['champion']['code_pins']:raise AgentError('Candidate changed during generalization')
 if g['status']=='passed':
  if hasattr(adapter,'publish_confirmed_patch'):adapter.publish_confirmed_patch(state,config,root)
  return True
 if g['status'] in ('failed','invalid'):state['status']='generalization_failed';return False
 index=g['case_index'];case=req['cases'][index];folder=root/'generalization'/req['id']/case['id']
 maker=getattr(adapter,'generalization_stage',None);reader=getattr(adapter,'generalization_result',None)
 if not case.get('ready') or not case.get('baseline') or maker is None or reader is None:
  state.update(status='waiting_generalization_assets',final_patch_status='pending_generalization',generalization_pending=True)
  write(root/'generalization_request.json',dict(request=g['request'],case=case,missing_reason=case.get('unavailable_reason','Registered baseline and execution adapter required'),confirmed=False));return False
 stage=maker(copy.deepcopy(case),copy.deepcopy(req['candidate']),folder,copy.deepcopy(config));verify(stage['admission'])
 admission=read(stage['admission']['path'])
 if admission.get('passed') is not True or admission.get('case_id')!=case['id'] or admission.get('candidate_pins')!=req['candidate']['code_pins'] or admission.get('baseline')!=case['baseline']:raise AgentError('Matched CPU/asset admission missing')
 inputs={**stage['inputs'],**req['candidate']['code_pins'],stage['admission']['path']:stage['admission']['sha256'],case['baseline']['path']:case['baseline']['sha256']}
 job=folder/'job'
 if not (job/'spec.json').exists() and not g.get('dispatch_reserved'):
  if state['usage'].get('gpu_attempts',0)>=config['budgets']['max_gpu_attempts']:raise BudgetExceeded('Generalization shares GPU budget')
  state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;state['usage']['gpu_dispatches']+=1;g['dispatch_reserved']=True
  write(root/'state.json',state)
 receipt=jobs.launch(job,stage['command'],stage.get('cwd',root),min(stage['timeout'],remaining(state,config)),inputs,stage['outputs'],Path(config.get('_host_stop_path',root/'STOP')));g['current_job']=str(job)
 if receipt['status']=='running':state['status']='running_generalization';return None
 if receipt['status']!='succeeded':
  g.update(status='invalid',failure=dict(case_id=case['id'],receipt_status=receipt['status'],reason='Incomplete infrastructure run; no confirmation'));state['status']='generalization_failed';write(folder/'gate_result.json',g['failure']);return False
 try:result=verify_case(reader(case,folder,receipt),req['candidate'],case,req['requested_epoch'],req['score_tolerance'])
 except (AgentError,OSError,KeyError,TypeError,ValueError) as e:
  g.update(status='invalid',failure=dict(case_id=case['id'],reason=str(e)));state['status']='generalization_failed';write(folder/'gate_result.json',g['failure']);return False
 write(folder/'gate_result.json',result);g['results'].append(result);g.update(case_index=index+1,current_job=None,dispatch_reserved=False)
 if not result['passed'] and not config.get('_host_collect_all_validation_cases'):g['status']='failed';state.update(status='generalization_failed',final_patch_status='rejected_generalization');return False
 if g['case_index']==len(req['cases']):
  if any(not x['passed'] for x in g['results']):g['status']='failed';state.update(status='generalization_failed',final_patch_status='rejected_generalization');return False
  finalize(state,req,root)
  if hasattr(adapter,'publish_confirmed_patch'):adapter.publish_confirmed_patch(state,config,root)
  return True
 state['status']='generalization_pending';return None
