"""Fresh actual probes. Candidate policy never receives offline edit/oracle truth."""

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

import argparse,copy,sys
pass  # Path is provided by the portable boundary above
from common_flow import *
sys.path.insert(0,str(DISCOVERY))
from episode import prepare,execute

def stage(taskfile,policyfile,work,case):
 task=read(taskfile);policy=read(policyfile);work=Path(work)
 if work.exists():
  verify_pins(read(work/'input_pins.json'));return
 base=Path(task['baseline_root'])
 new_edit=case=='cutin551'
 # Existing host generator audits dynamics and all other native actors.
 for receipt in task['edit_receipts'].values():verify(receipt)
 prepare(task['edit'],policy,work,baseline=None)
 source=Path(task['edit_assets']) if new_edit else Path(policy['release'])/'scenario'
 for f in ('scenario.json','target_track.npz'):(work/'scenario'/f).write_bytes((source/f).read_bytes())
 plan=read(work/'plan.json');plan['baseline_root']=str(base);plan['assets'].update(scenario=asset(work/'scenario/scenario.json'),target_track=asset(work/'scenario/target_track.npz'));write(work/'plan.json',plan)
 c=read(task['source_contracts']['cutin551' if case!='original551' else 'original551'])
 c.update(worker=sha(Path(policy['code'])/'worker.py'),scenario=sha(work/'scenario/scenario.json'),target_track=sha(work/'scenario/target_track.npz'),comparison_version=policy['version'],patch='on',fresh_history=True,score_kind='executed_path')
 c['source_assets'].update(scenario=sha(work/'scenario/scenario.json'),target_track=sha(work/'scenario/target_track.npz'))
 write(work/'contract.json',c)
 dc=read(work/'discovery_contract.json');dc.update(scenario=asset(work/'scenario/scenario.json'),track=asset(work/'scenario/target_track.npz'));write(work/'discovery_contract.json',dc)
 write(work/'input_pins.json',{str(p):sha(p) for p in work.rglob('*') if p.is_file() and p.name!='input_pins.json'})

def run(taskfile,policyfile,work,case):
 task=read(taskfile);policy=read(policyfile);work=Path(work);arm='A' if case=='original551' else 'B'
 stage(taskfile,policyfile,work,case)
 if (work/'completion.json').exists():
  from common_flow import metrics
  return metrics(work,arm)
 if (work/arm).exists():raise RuntimeError('Partial expensive episode retained: '+str(work))
 execute(task['edit'],policy,work,arm,mode='on')
 r=read(work/arm/'run_report.json');r.update(validation_contract=asset(work/'contract.json'),validation_patch='on',validation_case=case,comparison_kind='inherited_controller_vs_new_candidate')
 write(work/arm/'run_report.json',r)
 done=read(work/'completion.json');done.update(report=asset(work/arm/'run_report.json'),contract=asset(work/'contract.json'));write(work/'completion.json',done)
 from common_flow import metrics
 return metrics(work,arm)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute']);p.add_argument('--task',required=True);p.add_argument('--policy',required=True);p.add_argument('--work',required=True);p.add_argument('--case',choices=['original551','cutin551','legacy551'],required=True);a=p.parse_args()
 if a.stage=='prepare':stage(a.task,a.policy,a.work,a.case)
 else:run(a.task,a.policy,a.work,a.case)
