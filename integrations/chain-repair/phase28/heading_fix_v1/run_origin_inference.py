"""Replay saved inputs only; snapshot-matched inference ablations, no MPC."""

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

import datetime
import fcntl
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import shutil
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]
RUN=P28/'runs/heading_fix_scene552_v1'

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def original_request(index):
    request=_hugsim_expand(json.loads((RUN/f'uniad_worker/request_{index:03d}.json').read_text()))
    packet=RUN/f'uniad_worker/step_{index:03d}/observation.pkl'
    if request.get('index')!=index or Path(request['packet']).resolve()!=packet.resolve():
        raise ValueError('Original request does not match the saved frame')
    return request

def prepare_bootstrap_packet(job):
    """five_uniad.infer checks FIVE_JOB/observation.pkl before loading model."""
    source=Path(original_request(0)['packet']).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    destination=Path(job)/'step_000/observation.pkl'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.symlink_to(source)
    return destination

def wait(path,process,timeout=1200):
    start=time.monotonic();last=start
    while not path.exists():
        if process.poll() is not None:
            error_file=path.parent/'worker_error.json'
            error=_hugsim_expand(json.loads(error_file.read_text())).get('error') if error_file.is_file() else 'No structured worker error'
            raise RuntimeError(f'Worker exited ({process.returncode}): {error}; log: {path.parent.parent / "worker.log"}')
        now=time.monotonic()
        if now-start>timeout:raise TimeoutError(str(path))
        if now-last>30:print('Waiting:',path.name,round(now-start),'s',flush=True);last=now
        time.sleep(.2)
    result=_hugsim_expand(json.loads(path.read_text()))
    if result.get('status')=='error':raise RuntimeError(str(result))
    return result

def main():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run in GPU terminal')
    lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('Service STOP is present')
    if shutil.which('nvidia-smi'):
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(s.strip().isdigit() for s in active.splitlines()):raise RuntimeError('GPU has compute processes')
    # Keep the launch-pinned implementation intact; unrelated source edits
    # invalidate original-prefix comparisons and must not be silently accepted.
    manifest=_hugsim_expand(json.loads((HERE/'scene_launch_manifest.json').read_text()))['sha256']
    for name,digest in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+name)
    output=P28/'runs'/('candidate_origin_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(exist_ok=False);job=output/'worker';job.mkdir()
    bootstrap=prepare_bootstrap_packet(job)
    atomic(job/'origin_mode.json',dict(mode='baseline'))
    seed=original_request(0)['diagnostic_seed']
    if isinstance(seed,bool) or not isinstance(seed,int):raise ValueError('Invalid original diagnostic seed')
    env=dict(os.environ,BASE=str(ROOT),BEV_EXPERIMENT_SEED=str(seed),PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    python=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'
    print('OUTPUT:',output,flush=True)
    source_packets=[RUN/f'uniad_worker/step_{i:03d}/observation.pkl' for i in range(40)]
    packet_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_packets}
    code_files=[HERE/name for name in ('origin_worker.py','origin_state.py','run_origin_inference.py','analyze_candidate_origin.py')]
    code_files += [ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin'/name for name in (
        'uniad/detectors/uniad_track.py','uniad/detectors/uniad_e2e.py',
        'uniad/dense_heads/track_head_plugin/tracker.py','uniad/dense_heads/track_head_plugin/modules.py',
        'core/bbox/coders/detr3d_track_coder.py','uniad/scene_reliability.py','uniad/input_geometry.py')]
    code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    atomic(output/'diagnostic_manifest.json',dict(code_sha256=code_hashes,source_packet_sha256=packet_hashes,
        original_run=str(RUN),original_seed=seed,bootstrap_packet=str(bootstrap),
        old_snapshot_used=False,tracker_counter_in_new_snapshot=True))
    with (output/'worker.log').open('x') as log:
        process=subprocess.Popen([str(python),str(HERE/'origin_worker.py'),'--job',str(job),
            '--planning-bev','current','--model-profile','calibration_geometry_v1','--diagnostic-closed-loop'],
            env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            ready=wait(job/'ready.json',process)
            if ready['branch_restored'] is not None:raise RuntimeError('Diagnostic must start fresh; old snapshots omitted tracker counter')
            for index in range(39):
                atomic(job/f'request_{index:03d}.json',original_request(index))
                wait(job/f'done_{index:03d}.json',process,300)
                (job/f'request_{index:03d}.json').unlink()
                print('Replayed',index,flush=True)
            snapshot=output/'before39'
            atomic(job/'command_snapshot.json',dict(path=str(snapshot)))
            wait(job/'snapshot_done.json',process,300)
            cases={}
            for label,mode in [('baseline','baseline'),('drop_bev','drop_bev'),('drop_tracks','drop_tracks'),('drop_both','drop_both'),('baseline_repeat','baseline')]:
                if label!='baseline':
                    atomic(job/'command_restore.json',dict(path=str(snapshot)))
                    wait(job/'restore_done.json',process,300);(job/'restore_done.json').unlink()
                atomic(job/'origin_mode.json',dict(mode=mode))
                atomic(job/'request_039.json',original_request(39))
                wait(job/'done_039.json',process,300)
                (job/'request_039.json').unlink();(job/'done_039.json').unlink()
                case=output/label;shutil.move(str(job/'step_039'),str(case))
                with np.load(case/'perception.npz') as d:
                    boxes=d['candidate_boxes'].copy();scores=d['candidate_scores'].copy();labels=d['candidate_labels'].copy()
                with np.load(RUN/'uniad_worker/step_039/perception.npz') as d:
                    target=d['candidate_boxes'][33].copy()
                    errors={k:float(np.abs(loc-d[k]).max()) if loc.shape==d[k].shape else None for k,loc in [('candidate_boxes',boxes),('candidate_scores',scores),('candidate_labels',labels)]}
                close=np.flatnonzero((np.linalg.norm(boxes[:,:2]-target[:2],axis=1)<2)&(labels==0))
                query=_hugsim_expand(json.loads((case/'neural_query_origin.json').read_text()))
                rows=[dict(index=int(i),score=float(scores[i]),box=boxes[i].tolist(),neural_track_id=query['selected_track_ids'][i],
                    source_kind=query['selected_source_kind'][i],forward_input_index=query['selected_forward_input_indices'][i],
                    preupdate_track_id=query['selected_preupdate_track_ids'][i]) for i in close]
                cases[label]=dict(near_target_candidates=rows,maximum_array_errors_vs_original=errors,
                    special_ego_in_decode=query['decode_input_has_special_ego'])
                atomic(output/'report.json',dict(diagnostic_uncertified=True,formal_evaluation_admitted=False,cases=cases,
                    source_packet_sha256=packet_hashes,complete=False,
                    scope='Fresh replay frames0..38; same saved frame39. Restored identical pre39 state, tracker ID counter and RNG before each ablation, followed by baseline repeat. No rendering/control/MPC. Dropping tracking also changes active-query composition, not solely a learned memory tensor.'))
                print(label,cases[label],flush=True)
                if label in ('baseline','baseline_repeat') and any(v is None or v>1e-4 for v in errors.values()):
                    raise RuntimeError('Original baseline not reproduced; do not attribute ablations')
                if label=='baseline_repeat':
                    original_query=_hugsim_expand(json.loads((output/'baseline/neural_query_origin.json').read_text()))
                    if query['selected_track_ids']!=original_query['selected_track_ids']:
                        raise RuntimeError('Restored neural IDs differ; no clean causal comparison')
            for path,digest in packet_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
            for path,digest in code_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
            report=_hugsim_expand(json.loads((output/'report.json').read_text()));report['complete']=True;atomic(output/'report.json',report)
            from analyze_candidate_origin import analyze
            print(json.dumps(analyze(output),indent=2),flush=True)
        finally:
            atomic(job/'stop.json',{})
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    print('Completed:',output/'report.json',flush=True)

if __name__=='__main__':main()
