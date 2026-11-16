"""One pinned fresh counterfactual run, shared lock, automatic postprocessing."""

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

import fcntl
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]

def verify_pinned():
    pinned=_hugsim_expand(json.loads((HERE/'launch_manifest.json').read_text()))
    for name,digest in pinned.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+name)
    print('Pinned SHA256 verified:',len(pinned),'files',flush=True)

def main():
    verify_pinned()
    if sys.argv[1:]==['--check']:return
    request=_hugsim_expand(json.loads((HERE/'request.json').read_text()));run=Path(request['output'])
    if run.exists():raise RuntimeError('Refusing duplicate/overwrite: '+str(run))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run in GPU terminal; CUDA unavailable here')
    lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP is present')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip().isdigit() for s in active.splitlines()):raise RuntimeError('GPU already has compute processes')
    env=dict(os.environ,BASE=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    print('Archived prefix through step39; then live feedback with neural track92 removed. 117 steps; wall cap 18000s. OUTPUT:',run,flush=True)
    log_path=HERE/'simulation.log'
    with log_path.open('x') as log:
        process=subprocess.Popen([sys.executable,str(HERE/'run_harness.py'),str(HERE/'request.json')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:code=process.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    progress=run/'progress.json'
                    print(progress.read_text() if progress.exists() else 'Loading scene / UniAD; log: '+str(log_path),flush=True)
        except BaseException:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM)
            raise
    post=0
    if (run/'run_report.json').exists() and (run/'asset_reference.json').exists():
        with (HERE/'postprocess.log').open('x') as log:
            post=subprocess.call([sys.executable,str(HERE/'postprocess.py'),str(run)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('Simulation exit:',code,'postprocess exit:',post,'log:',log_path,flush=True)
    print('Inspect:',run/'analysis/summary.json',flush=True)
    if code or post:raise SystemExit(1)

if __name__=='__main__':main()
