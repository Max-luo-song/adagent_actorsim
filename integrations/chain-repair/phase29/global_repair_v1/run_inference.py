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
HERE=Path(__file__).resolve().parent;P28=HERE.parents[1]/'phase28';ROOT=P28.parents[5]
sys.path[:0]=[str(HERE.parent),str(P28),str(P28/'heading_fix_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
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
    from global_repair_v1.launch import verify_pinned
    return verify_pinned()

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
    output=HERE.parent/'runs'/('global_evidence_inference_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));output.mkdir(exist_ok=False)
    env=dict(os.environ,BASE=str(ROOT),BEV_EXPERIMENT_SEED=str(seed),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    python=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'
    report=dict(formal_evaluation_admitted=False,scope='81 saved observations per fresh arm; no rendering, MPC or control',source=str(SOURCE),seed=seed,complete=False,global_repair_complete=False,cases={})
    atomic(output/'report.json',report);print('OUTPUT:',output,flush=True)
    deadline=time.monotonic()+3600
    def remaining(limit):
        value=min(float(limit),deadline-time.monotonic())
        if value<=0:raise TimeoutError('Inference comparison wall cap 3600s')
        return value
    for mode in ('baseline','global'):
        job=output/mode;job.mkdir();(job/'step_000').mkdir();(job/'step_000/observation.pkl').symlink_to(SOURCE/'uniad_worker/step_000/observation.pkl')
        atomic(job/'global_mode.json',dict(mode=mode));summary=dict(frames=0,quarantined_observations=0,maximum_raw_array_error=0.,baseline_prefix_validated=False)
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
                        summary['maximum_raw_array_error']=max(summary['maximum_raw_array_error'],max(errors.values()))
                    if mode=='global':
                        with np.load(d/'perception.npz') as a,np.load(output/f'baseline/step_{i:03d}/perception.npz') as b:
                            raw_keys=('candidate_boxes','candidate_scores','candidate_labels','boxes','scores','labels','trajectory','occupancy')
                            for key in raw_keys:
                                if a[key].dtype!=b[key].dtype or not np.array_equal(a[key],b[key]):
                                    raise RuntimeError('Neural output changed under external-only admission: frame '+str(i)+' key '+key)
                    admission=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text())).get('proposal_admission',{'quarantined':[]})
                    summary['quarantined_observations']+=len(admission['quarantined'])
                    summary['frames']=i+1;report['cases'][mode]=summary;atomic(output/'report.json',report)
                    if i%10==0:print(mode,'frame',i,'quarantined',summary['quarantined_observations'],flush=True)
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
    from global_repair_v1.assess_inference import assess
    report['screening']=assess(output)
    report['normal_object_screen_passed']=report['screening']['visible_reference_changed_or_removed']==0
    report['raw_neural_arrays_identical']=True
    report['implementation_manifest_sha256']=verify();report['complete']=True;atomic(output/'report.json',report)
    print('Completed:',output/'report.json',flush=True)
if __name__=='__main__':main()
