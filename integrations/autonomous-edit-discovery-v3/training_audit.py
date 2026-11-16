"""Public robustness replay on training-scene actions only; no private gate data."""
import argparse,copy,sys
from common_flow import *
sys.path.insert(0,str(AGENT))
from core import AgentError,BudgetExceeded,digest,remaining
import jobs

def step(state,config,root):
 root=Path(root);policy=config.get('public_training_audit',{})
 if not policy.get('enabled') or not policy.get('suite'):state['status']='waiting_training_review_assets';return False
 verify(policy['suite']);suite=read(policy['suite']['path']);cases=[c for c in suite['cases'] if c.get('kind')=='action'][:2]
 if not cases or any(c['scene']!=config['generalization']['training_scene'] for c in cases):raise AgentError('Training audit cannot use a validation scene')
 folder=root/'training_stress'/digest(state['champion']['code_pins'])[:16];folder.mkdir(parents=True,exist_ok=True)
 request=folder/'request.json';expected=dict(candidate=state['champion'],cases=cases,training_scene=config['generalization']['training_scene'])
 if request.exists():
  if read(request)!=expected:raise AgentError('Frozen training audit changed')
 else:write(request,expected)
 result=folder/'results.json';job=folder/'job'
 if not state['blind_repair'].get('audit_reserved'):
  if state['usage'].get('gpu_attempts',0)>=config['budgets']['max_gpu_attempts']:raise BudgetExceeded('Public training audit shares GPU budget')
  # One reservation represents a bounded two-episode training audit job.
  state['usage']['gpu_attempts']=state['usage'].get('gpu_attempts',0)+1;state['usage']['gpu_dispatches']+=1;state['blind_repair']['audit_reserved']=True;write(root/'state.json',state)
 inputs={**state['champion']['code_pins'],str(request):sha(request),policy['suite']['path']:policy['suite']['sha256']}
 receipt=jobs.launch(job,['bash',str(LOCAL/'run.sh'),'_training_audit','--request',str(request),'--output',str(folder)],ROOT,min(config['budgets']['gpu_stage_seconds'],remaining(state,config)),inputs,[str(result)],root/'STOP')
 if receipt['status']=='running':state['status']='running_public_training_audit';state['public_training_audit_job']=str(job);return None
 if receipt['status']!='succeeded':state['status']='training_review_infrastructure_error';return False
 data=read(result)
 if data['candidate_pins']!=state['champion']['code_pins'] or data['training_scene']!=config['generalization']['training_scene']:raise AgentError('Training audit identity mismatch')
 for item in data['episodes']:
  for k in ('report','summary','contract'):verify(item[k])
 state['public_training_audit']=asset(result);state['blind_repair']['audit_required']=False;state['status']='ready';return True

def execute(request,output):
 import os
 os.fstat(9)
 import probe_worker
 request=read(request);output=Path(output);policy=request['candidate'];policyfile=output/'policy.json';write(policyfile,policy);episodes=[]
 for case in request['cases']:
  if case['scene']!=request['training_scene']:raise AgentError('Foreign scene in public training audit')
  verify(case['task']);work=output/case['id']
  episodes.append(probe_worker.run(case['task']['path'],policyfile,work,'cutin551'))
 write(output/'results.json',dict(candidate_pins=policy['code_pins'],training_scene=request['training_scene'],episodes=episodes,comparison_scope='unpaired public training stress observations only; not a patch gain or hidden validation result'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();execute(a.request,a.output)
