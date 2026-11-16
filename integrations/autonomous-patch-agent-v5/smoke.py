"""Exercise the real adapter through all CPU gates, with no model or GPU job."""

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

import datetime,json,sys,time
pass  # Path is provided by the portable boundary above
from core import LOCAL,ROOT,P26,AgentError,asset,pins,read,sha,write
from cli import configuration
from adapter import HUGSIMAdapter
import builder,jobs

def main():
 cfg=configuration(LOCAL/'config.example.json');adapter=HUGSIMAdapter();parent=adapter.bootstrap(cfg)
 if hasattr(adapter,'registered_pool'):
  from governance import select_parent
  pool=adapter.registered_pool(cfg,parent);parent=select_parent(pool,parent,parent['reference_metrics'])
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S').lower();rid='smoke_'+stamp
 rd=P26/'autonomous_patch_agent_v5/smoke'/rid;rd.mkdir(parents=True,exist_ok=False)
 empty={'patch_instruction':{'operations':[]}}
 bundle=builder.build(rd,rid,parent,empty,cfg,smoke=True);attempt=builder.attempt(rd,bundle,1)
 # Build source is frozen for each concrete job. This smoke is never a patch
 # decision and can never update the working champion or start a GPU episode.
 rows=[]
 for stage in adapter.stages(rd,bundle,attempt,cfg):
  if stage['name'] in ('causal_parent','causal_candidate'):continue
  if stage['gpu']:break
  inputs={**stage['inputs']}
  for own in ('jobs.py','core.py','probe.py','test_runner.py'):inputs[str(LOCAL/own)]=sha(LOCAL/own)
  for p in stage.get('deps',[]):inputs[str(Path(p).resolve())]=sha(p)
  folder=rd/'jobs'/stage['name'];r=jobs.launch(folder,stage['command'],ROOT,stage['timeout'],inputs,stage['outputs'],rd/'STOP')
  while r['status']=='running':time.sleep(.5);r=jobs.inspect(folder)
  if r['status']!='succeeded':raise AgentError('Real adapter smoke stage failed: '+stage['name']+' '+str(folder))
  adapter.validate_stage(stage['name'],rd,bundle,attempt);rows.append(dict(name=stage['name'],passed=True,receipt=asset(folder/'receipt.json')));print(stage['name'],'passed',flush=True)
 write(rd/'result.json',dict(passed=True,rows=rows,actual_registered_CPU_commands=True,model_API_called=False,GPU_executed=False,patch_applied=False,production_driver_changed=False,template_parent=parent['version']))
 print('SMOKE_OUTPUT',rd,flush=True)
if __name__=='__main__':main()
