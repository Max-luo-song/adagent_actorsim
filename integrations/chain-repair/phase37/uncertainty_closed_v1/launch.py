"""Unique attempt directories, pinned inputs, staged GPU probe then feedback."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,datetime,uuid

def verify():
 pins=_hugsim_expand(json.loads((HERE/'manifest.json').read_text()))
 for name,digest in pins.items():
  path=ROOT/name
  hasher=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''):hasher.update(block)
  if hasher.hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+str(path))
 print('Pinned SHA256 verified:',len(pins),'files',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 from diagnostic_admission import validate_launch
 validate_launch()
 verify()
 arm=_hugsim_expand(json.loads((HERE/'arm.json').read_text()))
 from protocol_harness import validate_snapshot_request
 validate_snapshot_request(_hugsim_expand(json.loads((HERE/'request_template.json').read_text())),arm['steps'])
 if args.check:return
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the GPU terminal')
 lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP present')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 if any(x.strip().isdigit() for x in active.splitlines()):raise RuntimeError('GPU already has compute processes')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=HERE.parent/'runs'/('uncertainty_closed551_'+stamp);out.mkdir(parents=True,exist_ok=False)
 request=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()));request['output']=str(out/'closed_loop');(out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
 print('OUTPUT:',out,flush=True);print('Scene551; 8m/s initial speed; stationary car ~45m ahead; current-frame neural reference; road soft weight10 plus distance; strategy 2m/s cap disabled; physical vehicle bounds retained; 30 controls / 3 seconds simulated; original 3 cameras; event MPC: current checks every0.1s, full replan at most0.5s apart or on events, original multistart by default; total wall cap7200s including probe/startup',flush=True)
 env=dict(os.environ,BASE=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');deadline=time.monotonic()+7200
 def stage(script,argv,name,cap=None):
  stage_deadline=min(deadline,time.monotonic()+cap) if cap else deadline
  with (out/(name+'.log')).open('x') as log:
   process=subprocess.Popen([sys.executable,str(HERE/script),*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while True:
     left=stage_deadline-time.monotonic()
     if left<=0:raise TimeoutError(name+' wall cap exceeded')
     try:return process.wait(timeout=min(30,left))
     except subprocess.TimeoutExpired:
      progress=out/'closed_loop/progress.json';print(progress.read_text() if progress.exists() else name+' running; log: '+str(out/(name+'.log')),flush=True)
   except BaseException:
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise
 try:
  code=stage('probe.py',[out/'edit_probe'],'edit_probe',1200)
  if code or not _hugsim_expand(json.loads((out/'edit_probe/report.json').read_text())).get('passed'):raise RuntimeError('Edit preview failed; inspect edit_probe.log; closed-loop not started')
  code=stage('run_driver.py',[out/'request.json'],'simulation');verify()
  from execution_reporting import finish_run
  result=finish_run(out,code,script=HERE/'postprocess.py',cwd=ROOT,env=env)
  raise SystemExit(result['exit_code'])
 except BaseException as exc:
  if not isinstance(exc,SystemExit) or exc.code:
   report_path=out/'closed_loop/run_report.json'
   if not (out/'launch_result.json').exists():
    from execution_reporting import finish_run
    finish_run(out,1,script=HERE/'postprocess.py',cwd=ROOT,env=env)
   (out/'error.json').write_text(json.dumps(dict(error=repr(exc),closed_loop_report_exists=report_path.exists(),simulation_log=str(out/'simulation.log')),indent=2)+'\n')
  raise
if __name__=='__main__':main()
