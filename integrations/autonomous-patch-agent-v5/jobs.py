"""Detached, journaled native jobs; resuming a coordinator never duplicates a live job."""

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

import argparse,os,signal,subprocess,sys,time
pass  # Path is provided by the portable boundary above
from core import AgentError,Deferred,asset,digest,event,process_identity,read,sha,utc,verify,verify_pins,write
_WORKERS=[]
def reap():
 global _WORKERS
 _WORKERS=[p for p in _WORKERS if p.poll() is None]

ENV_KEYS={'PATH','HOME','USER','LANG','LC_ALL','TMPDIR','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','CUDA_HOME','TORCH_EXTENSIONS_DIR'}
def environment():
 env={k:v for k,v in os.environ.items() if k in ENV_KEYS}
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 return env

def launch(folder,command,cwd,timeout,inputs,outputs,stop_path):
 folder=Path(folder);spec=dict(command=list(map(str,command)),cwd=str(cwd),timeout_seconds=float(timeout),inputs=inputs,outputs=list(map(str,outputs)),stop_path=str(stop_path),env=environment())
 # Timeout is frozen at first launch; a restarted coordinator must not extend it.
 folder.mkdir(parents=True,exist_ok=True);path=folder/'spec.json'
 if path.exists():
  old=read(path)
  if any(old[k]!=spec[k] for k in ('command','cwd','inputs','outputs','stop_path')):raise AgentError('Existing job specification differs')
  return inspect(folder)
 write(path,spec)
 with (folder/'worker.log').open('xb') as log:
  worker=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker',str(folder)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=environment())
 _WORKERS.append(worker);reap()
 write(folder/'owner.json',dict(pid=worker.pid,identity=process_identity(worker.pid),spec_sha256=sha(path),launched_at=utc()))
 return {'status':'running','folder':str(folder)}

def inspect(folder):
 reap()
 folder=Path(folder);receipt=folder/'receipt.json'
 if receipt.exists():
  r=read(receipt)
  if r['spec_sha256']!=sha(folder/'spec.json'):raise AgentError('Job specification changed')
  verify(r['log'])
  if r['status']=='succeeded':
   for a in r['outputs']:verify(a)
  return r
 for filename in ('running.json','worker_started.json','owner.json'):
  p=folder/filename
  if p.exists():
   meta=read(p)
   for pid_key,id_key in (('worker_pid','worker_identity'),('pid','identity'),('child_pid','child_identity')):
    pid=meta.get(pid_key);ident=meta.get(id_key)
    if pid and ident and process_identity(pid)==ident:return {'status':'running','folder':str(folder)}
 if (folder/'spec.json').exists() and time.time()-(folder/'spec.json').stat().st_mtime<5:
  return {'status':'running','folder':str(folder),'reason':'dispatch metadata settling'}
 # A dispatch can be interrupted between writing its spec and writing owner.
 # Never replay an ambiguous expensive stage in the same output directory.
 return {'status':'interrupted','folder':str(folder),'reason':'No live recorded process and no completion receipt'}

def cancel(folder):
 folder=Path(folder);write(folder/'CANCEL',dict(time=utc()))
 meta=folder/'running.json'
 if meta.exists():
  r=read(meta);pid=r.get('child_pid');ident=r.get('child_identity')
  if pid and ident and process_identity(pid)==ident:
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass

def worker(folder):
 folder=Path(folder);spec=read(folder/'spec.json');spec_hash=sha(folder/'spec.json');began=time.monotonic();log=folder/'output.log'
 result={'status':'failed','returncode':None,'outputs':[]}
 write(folder/'worker_started.json',dict(worker_pid=os.getpid(),worker_identity=process_identity(os.getpid()),spec_sha256=spec_hash,started_at=utc()))
 try:
  verify_pins(spec['inputs'])
  with log.open('xb') as stream:
   child=subprocess.Popen(spec['command'],cwd=spec['cwd'],env=spec['env'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
   write(folder/'running.json',dict(worker_pid=os.getpid(),worker_identity=process_identity(os.getpid()),child_pid=child.pid,child_identity=process_identity(child.pid),started_at=utc(),spec_sha256=spec_hash))
   reason=None
   while child.poll() is None:
    if Path(spec['stop_path']).exists() or (folder/'CANCEL').exists():reason='cancelled'
    elif time.monotonic()-began>=spec['timeout_seconds']:reason='timeout'
    elif log.stat().st_size>32*1024*1024:reason='log_limit'
    if reason:
     try:os.killpg(child.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:child.wait(timeout=3)
     except subprocess.TimeoutExpired:
      try:os.killpg(child.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      child.wait()
     break
    time.sleep(.2)
   if reason is None and (Path(spec['stop_path']).exists() or (folder/'CANCEL').exists()):reason='cancelled'
   result.update(status=reason or ('succeeded' if child.returncode==0 else 'failed'),returncode=child.returncode)
  if result['status']=='succeeded':
   result['outputs']=[asset(p) for p in spec['outputs']]
   verify_pins(spec['inputs'])
 except BaseException as e:
  # Trusted command errors only; API authorization never enters the job env.
  result.update(status='failed',error_type=type(e).__name__,error=str(e)[:2000])
  if not log.exists():log.write_text('Job failed before process start\n')
 result.update(spec_sha256=spec_hash,log=asset(log),seconds=time.monotonic()-began,finished_at=utc())
 write(folder/'receipt.json',result)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True);worker(parser.parse_args().worker)
