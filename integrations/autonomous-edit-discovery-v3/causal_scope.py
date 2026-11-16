"""Evidence-bound planner scope; certificate code remains immutable."""
import copy
from common_flow import *
from governance import PROFILES
PROFILE='v15_corridor'
PROTECTED=('JointReference.admit','JointReference.actor_certificate','JointReference.joint_certificate','JointReference.terminal_stop','JointReference.stop_states','JointReference._rollout_variant','JointReference.process','JointReference.actor_boxes','JointReference.update_tracks')

def runtime_spec(path):
 path=Path(path).resolve();spec=read(path);binding=OUT/'runtime_host_binding.json'
 if binding.exists():
  approved=read(binding)
  if approved.get('tasks',{}).get(str(path))==asset(path):
   verify(approved['revision']);revision=read(approved['revision']['path'])
   if approved['host_pins']!=revision['new_host_pins']:raise ValueError('Host binding differs from approved revision')
   verify_pins(approved['host_pins']);spec=copy.deepcopy(spec);spec['host_pins']=approved['host_pins'];spec['runtime_host_revision']=approved['revision'];spec['runtime_host_binding']=asset(binding);return spec
 verify_pins(spec['host_pins']);return spec

def prepare_scope(state,round,evidence,root):
 root=Path(root);task=state.get('active_task_id');grants=state.setdefault('repair_scope_grants',{})
 grant=grants.get(task)
 if grant:
  verify(grant);record=read(grant['path']);verify(record['terminal_receipt'])
 else:
  if round['parent'].get('profile') not in ('v15_joint',PROFILE):return evidence
  terminal=next((v for k,v in evidence['items'].items() if k.startswith('terminal_') and v.get('case')=='cutin551' and v.get('episode_role')=='working_champion'),None)
  if not terminal:return evidence
  facts=terminal['facts'];cert=facts.get('certificates',{}).get('nominal_certificate') or {}
  if facts.get('reason')!='no_jointly_verified_action' or not (cert.get('road') or {}).get('passed') or not cert.get('dynamics_bounds_passed') or (cert.get('actors') or {}).get('passed') is not False:return evidence
  verify(terminal['asset']);record=dict(profile=PROFILE,task_id=task,terminal_receipt=terminal['asset'],registered_methods=sorted(PROFILES[PROFILE]['methods']),protected_methods=list(PROTECTED),maximum_changed_methods=1,maximum_changed_lines=80,reason='Actual terminal actor conflict with no certified corridor; permit upstream candidate generation/speed selection, retain final validators',not_a_causal_proof_or_patch_success=True)
  file=root/'repair_scopes'/(task+'.json');write(file,record);grant=asset(file);grants[task]=grant
 if record['registered_methods']!=sorted(PROFILES[PROFILE]['methods']) or record['task_id']!=task:raise ValueError('Repair scope registration changed')
 round['parent']['profile']=PROFILE
 if state['champion']['code_pins']==round['parent']['code_pins']:state['champion']['profile']=PROFILE
 if round['parent'].get('pool_id') in state.get('registry',{}):state['registry'][round['parent']['pool_id']]['profile']=PROFILE
 evidence['selected_parent_profile']=PROFILE;evidence['items']['source']['editable_methods']=record['registered_methods']
 evidence['items']['repair_scope']={'asset':grant,'facts':record}
 return evidence
