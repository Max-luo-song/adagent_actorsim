"""Actual CPU-only preparation through the same task/probe adapters as GPU launch."""
import time
from common_flow import *
import flow,probe_worker

def run():
 audit=read(read(OUT/'checks/action_examples/latest.json')['path'])
 sample=audit['results'][0]
 if not sample['passed'] or not sample['offline_feasible_path']:raise ValueError('Example not admitted')
 work=OUT/'checks/action_transport'/str(time.time_ns());work.mkdir(parents=True)
 original=flow.OUT;flow.OUT=work
 try:
  policy=OUT.parent/'autonomous_edit_discovery_v2/global_v15.json'
  m=dict(policies={'stable':str(policy)},host_pins=flow.frozen_host())
  task=flow.task_prepare(m,dict(id='transport_cpu_only',phases=sample['phases']))
  spec=read(task);target=work/'probe'
  probe_worker.stage(task,policy,target,'cutin551')
  verify_pins(read(target/'input_pins.json'))
  config=read(target/'scenario/scenario.json')['config']
  if config['phases']!=sample['phases'] or spec['edit']['program']!='registered_native_action_sequence_v1':raise ValueError('Action plan lost in transport')
  if sha(target/'scenario/target_track.npz')!=spec['edit_receipts']['target_track.npz']['sha256']:raise ValueError('Probe did not receive exact generated trajectory')
  report=dict(passed=True,task=asset(task),contract=asset(target/'contract.json'),plan=asset(target/'plan.json'),exact_action_track_received=True,legacy_controller_unchanged=True,API_called=False,GPU_executed=False,controls_executed=False)
  write(work/'report.json',report);write(OUT/'checks/action_transport/latest.json',asset(work/'report.json'));print('Actual multi-stage task -> probe preparation passed; no GPU or controls')
 finally:flow.OUT=original
if __name__=='__main__':run()
