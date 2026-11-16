"""Pinned driver-only diagnostic; reuse the verified unchanged NN worker."""

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
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'code/HUGSIM').is_dir())
sys.path.insert(0,str(HERE.parent))

def verify_pinned():
    path=HERE/'launch_manifest.json';pinned=_hugsim_expand(json.loads(path.read_text()))
    for name,digest in pinned.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Pinned file changed: '+name)
    print('Pinned SHA256 verified:',len(pinned),'files',flush=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')

    parser.add_argument('--closed',type=Path,metavar='INFERENCE_REPORT')
    args=parser.parse_args();manifest=verify_pinned()
    if args.check:return
    from global_repair_v1.launch import verify_pinned as verify_nn,validate_inference as validate_nn
    source=args.closed or HERE.parent/'runs/global_evidence_inference_20261103T074943Z/report.json'
    validate_nn(_hugsim_expand(json.loads(source.read_text())),verify_nn())
    validation=_hugsim_expand(json.loads((HERE/'validation.json').read_text()))
    if not validation.get('full_saved_input_equivalence_passed') or not validation.get('independent_candidate_certificates_passed'):
        raise RuntimeError('Optimized MPC saved-input gate has not passed')
    print('Reusing validated unchanged v1 NN worker; bounded-unknown v2 policy with optimized MPC',flush=True)
    request=_hugsim_expand(json.loads((HERE/'request.json').read_text()));run=Path(request['output'])
    if run.exists():raise RuntimeError('Refusing to overwrite '+str(run))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the GPU terminal')
    lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP present')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip().isdigit() for s in active.splitlines()):raise RuntimeError('GPU already has compute processes')
    env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    print('Fresh feedback from step0; existing three cameras; support-bounded unknown policy and optimized MPC enabled; no fixed-ID removal; 117 steps; wall cap18000s. OUTPUT:',run,flush=True)
    log_path=HERE/'simulation.log'
    with log_path.open('x') as log:
        process=subprocess.Popen([sys.executable,str(HERE/'run_harness.py'),str(HERE/'request.json')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+float(request['termination']['max_wall_seconds'])
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Closed-loop wall cap reached, including startup')
                try:code=process.wait(timeout=min(30,remaining));break
                except subprocess.TimeoutExpired:
                    progress=run/'progress.json'
                    print(progress.read_text() if progress.exists() else 'Loading; log: '+str(log_path),flush=True)
        except BaseException:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM)
            raise
    verify_pinned()
    post=0
    if (run/'run_report.json').exists():
        post=subprocess.call([sys.executable,str(HERE/'postprocess.py'),str(run)],env=env,cwd=ROOT)
    print('Simulation exit:',code,'postprocess exit:',post,'log:',log_path,flush=True)
    raise SystemExit(code or post)

if __name__=='__main__':main()
