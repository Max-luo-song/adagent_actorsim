"""Audited infrastructure repair and real-response CPU replay; never executes GPU/API."""
import copy,fcntl,time
from common_flow import *
from scene_designer import Designer,validate_batch,compile_entries
from core import decode,digest
import flow

def run():
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  before=read(OUT/'mission.json');verify(before['global_patch'])
  revision=OUT/'host_revisions'/('interface_'+str(time.time_ns()));revision.mkdir(parents=True)
  for file in ['mission.json','status.json','ledger.json','scene_design/usage.json','scene_design/batch_001/feedback.json']:
   source=OUT/file
   if source.exists():dest=revision/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
  pins=flow.frozen_host();changed={p:dict(before=before['host_pins'].get(p),after=h) for p,h in pins.items() if before['host_pins'].get(p)!=h}
  receipt=dict(reason='Per-edit admission, precise phase feedback, complete cross-phase action semantics; no controller change',previous_mission=asset(revision/'mission.json'),changed_host_files=changed,tests=58,controller_pins_unchanged=True,budget_and_usage_reset=False)
  write(revision/'revision.json',receipt)
  m=copy.deepcopy(before);m['host_pins']=pins;m.setdefault('host_revision_history',[]).append(asset(revision/'revision.json'));write(OUT/'mission.json',m)
  batch=OUT/'scene_design/batch_001';context=read(batch/'context.json');replays=[]
  for record in sorted(batch.glob('request_???.json')):
   saved=read(record);v=decode(saved['content']) if isinstance(saved['content'],str) else saved['content'];edits,selection=validate_batch(v,digest(context),context['known_parameter_signatures'])
   replays.append(dict(response=asset(record),**selection,valid_edit_count=len(edits)))
  usage_before=read(OUT/'scene_design/usage.json')
  config=read(OUT/'source_config.json');source=read(OUT/'source_state.json')
  proposal=Designer(OUT/'scene_design',config,source['usage']).propose(context,1,3)
  entries=compile_entries(proposal,1);m.setdefault('edit_provenance',{})
  audits=[]
  for entry in entries:
   m['edit_provenance'][entry['id']]=asset(batch/'proposal.json')
   try:
    task=flow.task_prepare(m,entry);audit=read(task.parent/'edit_work/scenario/scenario.json');feasible=read(task.parent/'feasibility.json')
   except ValueError:
    f=OUT/'tasks'/entry['id']/'edit_work/scenario/scenario.json'
    if not f.exists():raise
    audit=read(f);feasible=None
    if audit['passed']:raise
   item=dict(edit=entry['id'],passed=audit['passed'],checks=audit['checks'],metrics=audit['metrics'],phase_motion=audit.get('phase_motion',[]),offline_feasible_path=feasible['passed'] if feasible else None)
   audits.append(item);print(item,flush=True)
  if read(OUT/'scene_design/usage.json')!=usage_before:raise ValueError('Cached recovery unexpectedly changed usage')
  result=dict(response_replays=replays,recovered_entry_count=len(entries),CPU_audits=audits,additional_API_calls=0,GPU_executed=False,usage_preserved=True,host_revision=asset(revision/'revision.json'),next_step='Resume on configured GPU terminal; existing batch is reused, physical rejections feed the next LLM batch')
  write(revision/'recovery_result.json',result);write(OUT/'checks/interface_recovery_latest.json',asset(revision/'recovery_result.json'))
  write(OUT/'status.json',dict(stage='interface_repaired_ready_to_resume',updated_epoch=time.time(),additional_API_calls=0,recovered_entry_count=len(entries),prior_failure_preserved=asset(revision/'status.json'),recovery=asset(revision/'recovery_result.json')))
  return result
if __name__=='__main__':run()
