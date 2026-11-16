"""GPU startup validation with a standard-library-only supervisor."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from startup_timing import atomic_json

BASE=Path(__file__).resolve().parents[1]


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


def wait(path,worker,seconds):
    start=time.monotonic()
    while True:
        if path.is_file():
            data=json.loads(path.read_text())
            if data.get('status')=='error':raise RuntimeError(str(data))
            return data
        error=path.parent/'worker_error.json'
        if error.is_file():raise RuntimeError(error.read_text())
        if worker.poll() is not None:raise RuntimeError('Worker exited before '+path.name+': '+str(worker.returncode))
        if time.monotonic()-start>seconds:raise TimeoutError('Worker response timeout: '+str(path))
        time.sleep(.5)


def run(request_path,output):
    request=json.loads(Path(request_path).read_text())
    if set(request)-{'schema','packet','runtime_bundle_manifest'} or not {'schema','packet'}<=set(request) or request['schema']!='uniad_startup_probe_v1':raise ValueError('Unexpected startup request')
    packet=Path(request['packet']).resolve()
    if BASE not in packet.parents or not packet.is_file():raise ValueError('Expected existing workspace observation')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    worker_folder=output/'worker';(worker_folder/'step_000').mkdir(parents=True)
    (worker_folder/'step_000/observation.pkl').symlink_to(packet)
    checkpoint=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    report=dict(schema='uniad_startup_probe_v1',status='running',training=False,n1_validation_suspended=True,
        formal_evaluation_admitted=False,closed_loop_run=False,control_actions_executed=False,
        packet=str(packet),planning_bev='current',model_profile='calibration_geometry_v1',initialization_budget_seconds=1200.)
    atomic_json(output/'report.json',report)
    atomic_json(output/'progress.json',dict(stage='auditing_frozen_checkpoint'))
    report['checkpoint_before']=fingerprint(checkpoint);report['packet_before']=fingerprint(packet)
    if request.get('runtime_bundle_manifest'):
        atomic_json(output/'progress.json',dict(stage='preparing_local_runtime_cache'))
        from runtime_bundle import prepare
        report['runtime_cache']=prepare(request['runtime_bundle_manifest'])
        atomic_json(output/'report.json',report)
    report['implementation_before']={str(p.relative_to(BASE)):fingerprint(p) for p in (
        BASE/'code/UniAD_SIM/tools/closeloop/e2e.py',BASE/'waymo23-native-import/five_uniad.py',
        BASE/'waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py',BASE/'waymo23-native-import/startup_timing.py',
        BASE/'waymo23-native-import/uniad_startup_probe.py',BASE/'waymo23-native-import/runtime_bundle.py')}
    atomic_json(output/'report.json',report)
    env=dict(os.environ,BASE=str(BASE),PYTHONUNBUFFERED='1',UNIAD_INIT_TRACE='1',
        UNIAD_STARTUP_REPORT=str(worker_folder/'startup.json'),BEV_EXPERIMENT_SEED='20261021',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',MPLCONFIGDIR=str(BASE/'outputs/gpu_test_service/matplotlib_cache'))
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    if report.get('runtime_cache'):env['PYTHONPATH']=report['runtime_cache']['site_packages']
    worker=None;started=time.monotonic();error=None
    try:
        with (output/'worker.log').open('w') as log:
            atomic_json(output/'progress.json',dict(stage='initializing_uniad'))
            worker=subprocess.Popen([str(BASE/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
                str(BASE/'waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py'),
                '--job',str(worker_folder),'--planning-bev','current','--model-profile','calibration_geometry_v1',
                '--diagnostic-closed-loop'],cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
            ready=wait(worker_folder/'ready.json',worker,1200.)
            report.update(ready=ready,spawn_to_ready_seconds=time.monotonic()-started)
            atomic_json(output/'report.json',report)
            atomic_json(output/'progress.json',dict(stage='validating_first_inference'))
            atomic_json(worker_folder/'request_000.json',dict(index=0,packet=str(packet),diagnostic_seed=20261021))
            result=wait(worker_folder/'done_000.json',worker,300.)
            if not Path(result['perception']).is_file():raise ValueError('No actual model output')
            report['first_inference']=result
            report['status']='startup_and_first_inference_passed'
    except BaseException as exc:
        error=exc;report.update(status='failed',error=repr(exc))
    finally:
        atomic_json(worker_folder/'stop.json',{})
        if worker is not None:
            try:worker.wait(timeout=20.)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:worker.wait(timeout=10.)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
        report['checkpoint_after']=fingerprint(checkpoint)
        report['checkpoint_unchanged']=report['checkpoint_before']==report['checkpoint_after']
        report['packet_unchanged']=report['packet_before']==fingerprint(packet)
        report['implementation_unchanged']=all(fingerprint(BASE/key)==value for key,value in report['implementation_before'].items())
        timing=worker_folder/'startup.json'
        if timing.is_file():
            report['startup']=json.loads(timing.read_text())
            ready_events=[e for e in report['startup']['events'] if e['stage']=='ready']
            if ready_events:report['optional_ui_modules_loaded']=ready_events[-1]['optional_ui_modules_loaded']
            cache_events=[e for e in report['startup']['events'] if e['stage']=='runtime_cache_ready']
            if cache_events:report['runtime_cache']=cache_events[-1]['runtime']
        if not (report['checkpoint_unchanged'] and report['packet_unchanged'] and report['implementation_unchanged']):
            error=RuntimeError('Frozen input/checkpoint/implementation changed');report.update(status='failed',error=repr(error))
        atomic_json(output/'report.json',report)
        atomic_json(output/'progress.json',dict(stage='failed' if error else 'completed'))
    if error:raise error
    print(json.dumps(dict(status=report['status'],spawn_to_ready_seconds=report['spawn_to_ready_seconds'],
        optional_ui_modules_loaded=report.get('optional_ui_modules_loaded')),indent=2),flush=True)
    return report
