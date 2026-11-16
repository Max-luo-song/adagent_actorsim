"""Two fresh, seed-matched saved-input replays. No rendering/MPC/control."""

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

import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(P28/'heading_fix_v1')]
from run_origin_inference import atomic
SOURCE=P28/'runs/lineage92_closed_scene552_v4'

def wait(path,process,timeout,log_path):
    started=time.monotonic();last=started
    while not path.exists():
        if process.poll() is not None:
            error_file=path.parent/'worker_error.json'
            error=_hugsim_expand(json.loads(error_file.read_text())).get('error') if error_file.exists() else 'No structured worker error'
            raise RuntimeError('Worker exited (%s): %s; log: %s'%(process.returncode,error,log_path))
        now=time.monotonic()
        if now-started>timeout:raise TimeoutError('Waiting for %s; log: %s'%(path,log_path))
        if now-last>=30:
            print('Waiting:',path.name,round(now-started),'s; log:',log_path,flush=True);last=now
        time.sleep(.2)
    result=_hugsim_expand(json.loads(path.read_text()))
    if result.get('status')=='error':raise RuntimeError('Worker request failed: %s; log: %s'%(result,log_path))
    return result

def verify():
    pinned=_hugsim_expand(json.loads((HERE/'launch_manifest.json').read_text()))
    for name,digest in pinned.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+name)
    print('Pinned SHA256 verified:',len(pinned),'files',flush=True)

def main():
    verify()
    if sys.argv[1:]==['--check']:return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run in GPU terminal')
    lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP present')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip().isdigit() for s in active.splitlines()):raise RuntimeError('GPU already has compute processes')
    request=_hugsim_expand(json.loads((SOURCE/'uniad_worker/request_000.json').read_text()));seed=request['diagnostic_seed']
    output=P28/'runs'/('proposal_evidence_inference_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));output.mkdir(exist_ok=False)
    env=dict(os.environ,BASE=str(ROOT),BEV_EXPERIMENT_SEED=str(seed),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    python=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'
    report=dict(formal_evaluation_admitted=False,scope='81 saved observations per fresh arm; no rendering, MPC or control',source=str(SOURCE),seed=seed,complete=False,cases={})
    atomic(output/'report.json',report);print('OUTPUT:',output,flush=True)
    deadline=time.monotonic()+3600
    def remaining(limit):
        value=min(float(limit),deadline-time.monotonic())
        if value<=0:raise TimeoutError('Inference comparison wall cap 3600s')
        return value
    for mode in ('baseline','repair'):
        job=output/mode;job.mkdir();(job/'step_000').mkdir();(job/'step_000/observation.pkl').symlink_to(SOURCE/'uniad_worker/step_000/observation.pkl')
        atomic(job/'lifecycle_mode.json',dict(mode=mode));summary=dict(frames=0,cache_rows_retired=0,retirement_events=[],maximum_baseline_array_error=0.,baseline_prefix_validated=False)
        report['cases'][mode]=summary;atomic(output/'report.json',report)
        log_path=output/f'{mode}.log'
        with log_path.open('x') as log:
            process=subprocess.Popen([str(python),str(HERE/'worker.py'),'--job',str(job),'--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                ready=wait(job/'ready.json',process,remaining(1200),log_path)
                if ready['branch_restored'] is not None:raise RuntimeError('Fresh worker required')
                for i in range(81):
                    source=SOURCE/f'uniad_worker/step_{i:03d}';d=job/f'step_{i:03d}';d.mkdir(exist_ok=True)
                    packet=d/'observation.pkl'
                    if not packet.exists():packet.symlink_to(source/'observation.pkl')
                    req=dict(index=i,packet=str(packet))
                    if i==0:req['diagnostic_seed']=seed
                    atomic(job/f'request_{i:03d}.json',req);wait(job/f'done_{i:03d}.json',process,remaining(300),log_path);(job/f'request_{i:03d}.json').unlink()
                    with np.load(d/'perception.npz') as a,np.load(source/'perception.npz') as b:
                        errors={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) if a[k].shape==b[k].shape else None for k in ('candidate_boxes','candidate_scores','candidate_labels','trajectory','occupancy')}
                    if mode=='baseline':
                        if any(v is None or v>1e-4 for v in errors.values()):raise RuntimeError('Baseline arrays not reproduced at '+str(i)+': '+str(errors))
                        summary['maximum_baseline_array_error']=max(summary['maximum_baseline_array_error'],max(errors.values()))
                    transition=_hugsim_expand(json.loads((d/'query_lifecycle_transition.json').read_text()))
                    removed=transition['before_rows']-transition['after_rows'];summary['cache_rows_retired']+=removed
                    if removed:summary['retirement_events'].append(dict(step=i,removed=removed,ids=transition['retired_previous_cache_ids']))
                    summary['frames']=i+1;report['cases'][mode]=summary;atomic(output/'report.json',report)
                    if i%10==0:print(mode,'frame',i,'cache retired',summary['cache_rows_retired'],flush=True)
                summary['baseline_prefix_validated']=mode=='baseline'
            except BaseException as exc:
                report.update(failed=True,error=repr(exc),failure_case=mode,worker_log=str(log_path))
                atomic(output/'report.json',report)
                raise
            finally:
                atomic(job/'stop.json',{})
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        report['cases'][mode]=summary;atomic(output/'report.json',report)
    from assess import assess
    report['screening']=assess(output)
    verify();report['complete']=True;atomic(output/'report.json',report)
    print('Completed:',output/'report.json',flush=True)
if __name__=='__main__':main()
