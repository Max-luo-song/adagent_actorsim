
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

import sys,subprocess,json,time
pass  # Path is provided by the portable boundary above
p=Path(__file__).resolve().parent;root=next(x for x in p.parents if (x/'code/HUGSIM').is_dir());old=_hugsim_expand(json.loads((p/'matched_replay_summary.json').read_text()));rows=[]
def run(script,args,name):
 (p/'verification_progress.json').write_text(json.dumps({'stage':name,'new_gpu_executed':False})+'\n')
 with (p/(name+'.log')).open('w') as log:r=subprocess.run([sys.executable,'-B',str(p/script),*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 if r.returncode:raise RuntimeError('Failed '+name+'; inspect its log')
 print(name,'PASS',flush=True)
for index in [46,100]:
 run('replay_current.py',['--step',str(index)],'verified_after_%03d'%index);summary=_hugsim_expand(json.loads((p/('current_replays/step_%03d_summary.json'%index)).read_text()));result=_hugsim_expand(json.loads((p/('current_replays/step_%03d_result.json'%index)).read_text()));before=next(x for x in old if x['step']==index);cache=[x['caches'].get('clear',{}) for x in result['performance_cache_layers']];rows.append({'step':index,'before_seconds':before['before_seconds'],'after_seconds':summary['wall_seconds'],'speedup':before['before_seconds']/summary['wall_seconds'],'before':before['before'],'after':summary['after'],'shared_clear_cache':cache,'new_gpu_executed':False});(p/'matched_replay_summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
run('independent_current_check.py',[],'independent_current_check');run('replay_dropout.py',[],'actual_dropout_check');run('replay_agentless.py',[],'actual_agentless_check');(p/'verification_progress.json').write_text(json.dumps({'stage':'cpu_replays_completed','new_gpu_executed':False})+'\n')
