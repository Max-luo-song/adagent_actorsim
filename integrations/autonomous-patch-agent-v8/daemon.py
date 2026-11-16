"""Detach only the coordinator; credentials stay in its environment, never specs."""

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

import os,subprocess
pass  # Path is provided by the portable boundary above
from core import Deferred,lock,process_identity,read,sha,utc,write
_HANDLES=[]
def active(root):
 p=Path(root)/'coordinator_owner.json'
 if not p.exists():return None
 owner=read(p);pid=owner.get('pid');identity=owner.get('identity')
 return owner if pid and identity and process_identity(pid)==identity else None

def launch(root,command,cwd,env=None):
 root=Path(root)
 with lock(root/'start.lock'):
  previous=active(root)
  if previous:return dict(status='already_running',pid=previous['pid'])
  with (root/'coordinator.log').open('ab') as stream:
   process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  _HANDLES.append(process)
  owner=dict(pid=process.pid,identity=process_identity(process.pid),started_at=utc(),command=list(map(str,command)),credentials_in_environment_only=True)
  write(root/'coordinator_owner.json',owner);return dict(status='started',pid=process.pid,log=str(root/'coordinator.log'))

def summary(state,root):
 root=Path(root);r=state.get('active_round') or {};owner=active(root)
 result={k:v for k,v in state.items() if k not in ('champion','reference_metrics','history','active_round')}
 result.update(campaign_directory=str(root),champion_version=state['champion']['version'],champion_scores={case:item['score'] for case,item in state['champion']['metrics'].items()},round_id=r.get('id'),round_phase=r.get('phase'),current_job=r.get('current_job'),history_count=len(state['history']),coordinator_alive=owner is not None,coordinator_pid=owner['pid'] if owner else None)
 if 'registry' in state:
  from governance import context
  result['iteration_context']=context(state['registry'],state['champion'],state['reference_metrics'])
 progress={};out=(r.get('attempt') or {}).get('output')
 if out:
  for case,arm in [('original551','A'),('cutin551','B')]:
   p=Path(out)/f'cases/{case}/on/{arm}/progress.json'
   if p.exists():
    try:progress[case]=read(p)
    except (OSError,ValueError):progress[case]={'stage':'progress update in flight'}
 if state.get('active_task_id'):
  task=state['active_task_id'];ledger=state['task_convergence'][task];result['task_progress']={'task_id':task,'small_gain_streak':ledger['small_gain_streak'],'window':ledger['policy']['window'],'minimum_gain':ledger['policy']['minimum_gain'],'converged':ledger['converged']}
 result['episode_progress']=progress
 if r.get('phase')=='propose':
  path=root/'rounds'/r['id']/'model_calls'/('request_'+str(r.get('model_attempts',0)).zfill(3)+'.progress.json')
  if path.exists():
   try:result['api_progress']=read(path)
   except (OSError,ValueError):pass
 return result
