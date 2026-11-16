"""Live operator entry; API credentials are read only from environment."""

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

import argparse,ast,json,os,subprocess,sys
pass  # Path is provided by the portable boundary above
from core import AgentError,Deferred,LOCAL,ROOT,P26,identity,read,write,utc
from contracts import ALLOWED_METHODS
from engine import Engine
from adapter import HUGSIMAdapter
from provider import HTTPProvider

def configuration(path):
 c=read(path)
 if c.get('schema')!='autonomous_patch_config_v1' or set(c['editable_methods'])!=ALLOWED_METHODS:raise AgentError('Unknown configuration or unregistered edit scope')
 for k,v in c['budgets'].items():
  if not isinstance(v,(int,float)) or isinstance(v,bool) or v<=0:raise AgentError('Budgets must be positive: '+k)
 for k in ('max_output_tokens','timeout_seconds','max_response_bytes'):
  if not isinstance(c['api'][k],(int,float)) or isinstance(c['api'][k],bool) or c['api'][k]<=0:raise AgentError('API bounds must be positive: '+k)
 if c['api']['token_parameter'] not in ('max_tokens','max_completion_tokens') or c['api']['response_format'] not in ('text','json_object','json_schema'):raise AgentError('Unsupported API protocol option')
 if c['budgets']['poll_seconds']>30 or c['api']['timeout_seconds']>600:raise AgentError('Polling/API timeout too large')
 if c['context']['max_chars']<20000 or c['context']['max_chars']>500000:raise AgentError('Context budget outside supported bounds')
 # These are fixed engineering criteria for this registered adapter.
 if c['acceptance']!=read(LOCAL/'config.example.json')['acceptance']:raise AgentError('Acceptance criteria may not be weakened in an existing adapter')
 return c

def doctor(c,deep=False):
 result=dict(api_ready=not HTTPProvider(c['api']).readiness(),missing_api_variables=HTTPProvider(c['api']).readiness(),GPU_run_started=False)
 champion=HUGSIMAdapter().bootstrap(c);result.update(seed_version=champion['version'],seed_target_HDScore=champion['metrics']['cutin551']['score']['hdscore'],seed_original_HDScore=champion['metrics']['original551']['score']['hdscore'],held_out_scenes_registered=False,global_patch_validated=False)
 if c.get('candidate_registry'):
  from governance import initial_pool,context
  pool=initial_pool(c,champion);result['iteration_context']=context(pool,champion,champion['reference_metrics'])
 if deep:
  import tempfile
  with tempfile.TemporaryDirectory(prefix='patch_doctor_') as folder:
   output=Path(folder)/'smpl.json';r=subprocess.run([sys.executable,'-B',str(LOCAL/'probe.py'),'--code',c['seed']['code'],'--output',str(output)],cwd=ROOT,env=__import__('jobs').environment(),capture_output=True,text=True)
   if r.returncode:raise AgentError('Actual SMPL preflight failed: '+r.stdout[-1500:]+r.stderr[-1500:])
   result['smpl_preflight']=read(output)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','run','step','resume','start','status','stop','doctor','selftest']);p.add_argument('--campaign',default='patch_auto_v2');p.add_argument('--config',type=Path,default=LOCAL/'config.example.json');p.add_argument('--deep',action='store_true');a=p.parse_args()
 try:
  name=identity(a.campaign);root=P26/'autonomous_patch_agent_v2/campaigns'/name
  if a.command=='selftest':
   raise SystemExit(subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(LOCAL/'tests'),'-v'],cwd=ROOT).returncode)
  if a.command=='doctor':out=doctor(configuration(a.config),a.deep)
  elif a.command=='init':out=Engine(root,configuration(a.config)).initialize(name)
  elif a.command=='stop':
   if not (root/'state.json').exists():raise AgentError('Campaign has not been initialized')
   write(root/'STOP',dict(time=utc(),reason='operator stop'));out={'stop_requested':True,'campaign':name,'note':'Registered jobs cancel their own process groups; partial GPU artifacts are retained'}
  elif a.command=='status':out=read(root/'state.json')
  elif a.command=='start':
   from daemon import launch
   state=read(root/'state.json');c=read(root/'config.json');missing=HTTPProvider(c['api']).readiness()
   if missing:out={'status':'waiting_api','missing_api_variables':missing,'coordinator_started':False}
   elif (root/'STOP').exists() or state['status'] in ('budget_exhausted','blocked_infrastructure','model_stopped'):out={'status':state['status'],'coordinator_started':False,'note':'Use resume for operator stop; use a new campaign after exhausted budgets'}
   else:out=launch(root,[sys.executable,'-u','-B',str(LOCAL/'cli.py'),'run','--campaign',name],ROOT,dict(os.environ))
  else:
   if not (root/'state.json').exists():raise AgentError('Run init first; API configuration can remain empty')
   if a.command=='resume':
    stop=root/'STOP'
    if stop.exists():stop.unlink()
    state=read(root/'state.json')
    if state['status'] in ('stopped','coordinator_detached','deferred','waiting_api','integrity_or_executor_error'):state['status']='ready';write(root/'state.json',state)
   out=Engine(root).run(one_step=a.command=='step')
  if isinstance(out,dict) and 'champion' in out:
   from daemon import summary
   out=summary(out,root)
  print(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False))
 except (AgentError,OSError,KeyError,ValueError) as e:
  print(json.dumps({'status':'not_started_or_blocked','error':str(e)},ensure_ascii=False));raise SystemExit(2)
if __name__=='__main__':main()
