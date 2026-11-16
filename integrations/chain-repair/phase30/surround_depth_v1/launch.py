"""Pinned two-stage GPU experiment: sensor/motion gate, then fresh closed-loop."""

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

import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];P29=CR/'phase29'
ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path[:0]=[str(HERE.parent),str(P29)]
def verify_pinned():
 path=HERE/'launch_manifest.json';manifest=_hugsim_expand(json.loads(path.read_text()))
 for name,digest in manifest.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+name)
 print('Pinned SHA256 verified:',len(manifest),'files',flush=True)
 return hashlib.sha256(path.read_bytes()).hexdigest()
def stop_child(process):
 if process.poll() is not None:return
 for signum,seconds in ((signal.SIGINT,15),(signal.SIGTERM,10),(signal.SIGKILL,5)):
  try:os.killpg(process.pid,signum)
  except ProcessLookupError:return
  try:process.wait(timeout=seconds);return
  except subprocess.TimeoutExpired:pass
 raise RuntimeError('GPU child did not exit')
def run_child(arguments,log_path,progress,deadline,env):
 with log_path.open('x') as log:
  process=subprocess.Popen([sys.executable,*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while True:
    left=deadline-time.monotonic()
    if left<=0:raise TimeoutError('Experiment wall cap, including GPU startup')
    try:return process.wait(timeout=min(30,left))
    except subprocess.TimeoutExpired:print(progress.read_text() if progress.exists() else 'Loading; log: '+str(log_path),flush=True)
  except BaseException:
   stop_child(process);raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 manifest=verify_pinned();validation=_hugsim_expand(json.loads((HERE/'validation.json').read_text()))
 if not validation.get('cpu_gates_passed'):raise RuntimeError('CPU integration gate incomplete')
 if args.check:return
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this terminal; run this pinned script in the GPU terminal')
 request=_hugsim_expand(json.loads((HERE/'request.json').read_text()));probe=_hugsim_expand(json.loads((HERE/'probe.json').read_text()));run=Path(request['output']);probe_output=Path(probe['output'])
 for path in (run,probe_output,HERE/'probe.log',HERE/'simulation.log'):
  if path.exists():raise RuntimeError('Refusing to overwrite '+str(path))
 from global_repair_v1.launch import validate_inference,verify_pinned as verify_nn
 source=P29/'runs/global_evidence_inference_20261103T074943Z/report.json';validate_inference(_hugsim_expand(json.loads(source.read_text())),verify_nn())
 lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP present')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 if any(s.strip().isdigit() for s in active.splitlines()):raise RuntimeError('GPU already has compute processes')
 env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',SURROUND_LAUNCH_MANIFEST_SHA256=manifest)
 deadline=time.monotonic()+float(request['termination']['max_wall_seconds'])
 print('Stage 1: supplemental depth/motion probe on saved poses. No NN inference or new control. OUTPUT:',probe_output,flush=True)
 code=run_child([str(HERE/'sensor_probe.py')],HERE/'probe.log',probe_output/'progress.json',min(deadline,time.monotonic()+float(probe['maximum_wall_seconds'])),env)
 if code:raise RuntimeError('Sensor probe failed (exit %s); inspect %s; closed-loop not started'%(code,HERE/'probe.log'))
 result=_hugsim_expand(json.loads((probe_output/'report.json').read_text()))
 if not result.get('passed') or not result.get('nn_packet_unchanged') or not result.get('front_rgb_depth_restoration_exact') or result.get('launch_manifest_sha256')!=manifest:raise RuntimeError('Sensor probe report did not pass provenance/observation gate')
 verify_pinned()
 print('Stage 2: fresh closed-loop, three frozen UniAD RGB views plus four external depth views; sensor-estimated dynamic constraints; optimized MPC. 117 steps; total GPU wall cap18000s. OUTPUT:',run,flush=True)
 try:
  code=run_child([str(HERE/'run_harness.py'),str(HERE/'request.json')],HERE/'simulation.log',run/'progress.json',deadline,env)
 except BaseException:
  if (run/'run_report.json').exists():subprocess.call([sys.executable,str(HERE/'postprocess.py'),str(run)],env=env,cwd=ROOT)
  raise
 verify_pinned();post=0
 if (run/'run_report.json').exists():post=subprocess.call([sys.executable,str(HERE/'postprocess.py'),str(run)],env=env,cwd=ROOT)
 print('Simulation exit:',code,'postprocess exit:',post,'log:',HERE/'simulation.log',flush=True)
 raise SystemExit(code or post)
if __name__=='__main__':main()
