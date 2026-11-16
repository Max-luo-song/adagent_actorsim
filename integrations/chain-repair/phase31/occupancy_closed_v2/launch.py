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
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();verify()
 validation=_hugsim_expand(json.loads((HERE/'validation.json').read_text()))
 required=('regression_tests','typed_transport_tests','resident_worker_export_tests','continuous_loss_and_recovery_tests','driver_runtime_gate','legacy_worker_import','solver_and_geometry_regression','strategy_speed_policy','independent_speed_geometry')
 if not validation.get('passed') or not all(validation.get('required_checks',{}).get(k) is True for k in required):raise RuntimeError('Complete occupancy CPU preflight has not passed')
 if args.check:return
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the GPU terminal')
 lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP present')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 if any(x.strip().isdigit() for x in active.splitlines()):raise RuntimeError('GPU already has compute processes')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=HERE.parent/'runs'/('occupancy_closed551_'+stamp);out.mkdir(parents=True,exist_ok=False)
 request=_hugsim_expand(json.loads((HERE/'request_template.json').read_text()));request['output']=str(out/'closed_loop');(out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
 print('OUTPUT:',out,flush=True);print('Scene551; 8m/s initial speed; stationary car ~45m ahead; strategy 2m/s cap disabled; physical vehicle bounds retained; 197 controls; original 3 cameras; optimized MPC; total wall cap18000s including probe/startup',flush=True)
 env=dict(os.environ,BASE=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');deadline=time.monotonic()+18000
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
  code=stage('run_driver.py',[out/'request.json'],'simulation');verify();post=0
  if (out/'closed_loop/run_report.json').exists():post=subprocess.call([sys.executable,str(HERE/'postprocess.py'),str(out/'closed_loop')],env=env,cwd=ROOT)
  print('Simulation exit:',code,'postprocess exit:',post,'summary:',out/'closed_loop/analysis/summary.json',flush=True)
  raise SystemExit(code or post)
 except BaseException as exc:
  if not isinstance(exc,SystemExit) or exc.code:
   (out/'error.json').write_text(json.dumps(dict(error=repr(exc),closed_loop_report_exists=(out/'closed_loop/run_report.json').exists()),indent=2)+'\n')
  raise
if __name__=='__main__':main()
