"""Same saved inputs and CPU allocation, before/after, no GPU inference."""

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

import os,sys,json,signal,subprocess,time
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=next(x for x in HERE.parents if (x/'code/HUGSIM').is_dir());baseline=HERE.parent/'native_control_baseline_v1';bundle=CR/'phase33/scene_data551_v1/runs/scene551_20261106T050218Z';paused=None;rows=[]
status=bundle/'pipeline_status.json'
if status.exists():
 data=_hugsim_expand(json.loads(status.read_text()));pid=int(data['pid']);proc=Path('/proc')/str(pid)
 if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
  command=(proc/'cmdline').read_bytes()
  if b'pipeline.py' in command and str(bundle).encode() in command:os.kill(pid,signal.SIGSTOP);paused=pid;print('Semantic CPU process paused for matched measurement',pid,flush=True)
try:
 for index in [46,100]:
  for name,package in [('before',baseline),('after',HERE)]:
   with (HERE/('matched_%s_%03d.log'%(name,index))).open('w') as log:
    r=subprocess.run([sys.executable,'-B',str(package/'replay_current.py'),'--step',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
   if r.returncode:raise RuntimeError('Replay failed: %s step%d'%(name,index))
   print(name,'step',index,'completed',flush=True)
  before=_hugsim_expand(json.loads((baseline/('current_replays/step_%03d_summary.json'%index)).read_text()));after=_hugsim_expand(json.loads((HERE/('current_replays/step_%03d_summary.json'%index)).read_text()));raw=_hugsim_expand(json.loads((HERE/('current_replays/step_%03d_result.json'%index)).read_text()));cache=[x['caches'].get('clear',{}) for x in raw.get('performance_cache_layers',[])];rows.append({'step':index,'before_seconds':before['wall_seconds'],'after_seconds':after['wall_seconds'],'speedup':before['wall_seconds']/after['wall_seconds'],'before':before['after'],'after':after['after'],'shared_clear_cache':cache,'new_gpu_executed':False});(HERE/'matched_replay_summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
finally:
 if paused is not None:
  try:os.kill(paused,signal.SIGCONT);print('Semantic CPU process resumed',paused,flush=True)
  except ProcessLookupError:pass
