"""Start one recoverable detached CPU-only data preparation process."""

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

import argparse,json,os,subprocess,sys
pass  # Path is provided by the portable boundary above
from scene_data import write_json
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();bundle=a.bundle.resolve();status=bundle/'pipeline_status.json'
if status.exists():
 old=_hugsim_expand(json.loads(status.read_text()))
 proc=Path('/proc')/str(int(old['pid']))
 if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':
  command=(proc/'cmdline').read_bytes()
  if b'pipeline.py' in command and str(bundle).encode() in command:raise RuntimeError('Data preparation already running: '+str(old['pid']))
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
for key in ['LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']:env.pop(key,None)
log=bundle/'pipeline.log'
with log.open('ab') as stream:
 process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('pipeline.py')),str(bundle)],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
write_json(bundle/'pipeline_launch.json',{'pid':process.pid,'log':str(log),'cpu_threads':4,'simulation_started':False});print(json.dumps({'pid':process.pid,'log':str(log),'status':str(status)},indent=2),flush=True)
