"""Durable task identity and one shared GPU lock; never resume ambiguous controls."""

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

import contextlib,fcntl,os,subprocess,sys
pass  # Path is provided by the portable boundary above
from platform_io import ROOT,CR,LOCAL,DEFAULT_STORE,read,write,asset,fingerprint,identity,verify
from platform_catalog import Catalog
from platform_contracts import compile_task
PYTHON=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'

class Runner:
    def __init__(self,store=DEFAULT_STORE,catalog=None):self.store=Path(store);self.catalog=catalog or Catalog()
    def prepare(self,task,request_id):
        identity(request_id);compiled=compile_task(task,self.catalog);task=compiled['task'];digest=fingerprint(task)
        job=self.store/'jobs'/request_id;job.parent.mkdir(parents=True,exist_ok=True)
        # Atomic mkdir serializes first writer; the short store lock handles concurrent prepares.
        with (job.parent/'.prepare.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if job.exists():
                if not (job/'manifest.json').exists():raise RuntimeError('Incomplete preparation retained: '+str(job))
                if read(job/'manifest.json')['task_digest']!=digest:raise ValueError('Request ID reused with different task')
                return job
            profile=self.catalog.verify(compiled['profile_id']);job.mkdir()
            write(job/'task.json',task);write(job/'compiled_task.json',compiled);write(job/'catalog.json',self.catalog.data)
            write(job/'manifest.json',dict(schema='scene_platform_job_v1',request_id=request_id,task_digest=digest,
                task=asset(job/'task.json'),catalog=asset(job/'catalog.json'),compiled_task=asset(job/'compiled_task.json'),
                platform_code=[asset(p) for p in sorted(LOCAL.glob('*.py'))],driver_profile=compiled['profile_id']))
            write(job/'state.json',dict(state='prepared',controls_started=False))
        return job
    def _child(self,stage,job):
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        native=str(CR/'phase46/cutin551_white_edge_v1/native_runtime/lib')
        env['LD_LIBRARY_PATH']=native+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (job/(stage+'.log')).open('a') as log:
            code=subprocess.run([str(PYTHON),'-u','-B',str(LOCAL/'driver_worker.py'),stage,str(job)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('Platform '+stage+' failed; inspect '+str(job/(stage+'.log')))
    def check(self,task,request_id):
        job=self.prepare(task,request_id);self._verify(job);self._child('check',job);return dict(job=str(job),**read(job/'preflight.json'))
    def _verify(self,job,*,execution=True):
        manifest=read(job/'manifest.json')
        for k in ['task','catalog','compiled_task']:verify(manifest[k])
        if execution:
            for a in manifest['platform_code']:verify(a)
            Catalog(job/'catalog.json').verify(manifest['driver_profile'])
    def run(self,task,request_id):
        return self._execute(task,request_id,'run')
    def sensor_validate(self,task,request_id):
        # Independent diagnostic namespace: cannot be confused with driving execution.
        return Runner(self.store/'sensor_validation',self.catalog)._execute(task,request_id,'sensor-validate')
    def _execute(self,task,request_id,stage):
        job=self.prepare(task,request_id)
        with (job/'execution.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('This request is already running; inspect '+str(job/'state.json'))
            state=read(job/'state.json')
            if state['state'] in ('completed','failed','terminated') and (job/'task_result.json').exists():
                self._verify(job,execution=False);verify(state['result']);result=read(job/'task_result.json')
                for item in result.get('artifacts',[]):
                    if isinstance(item,dict) and 'path' in item and 'sha256' in item:verify(item)
                return dict(result=result,replayed=True,job=str(job))
            self._verify(job)
            if state['state']!='prepared':raise RuntimeError('Interrupted task retained; inspect before explicit recovery, no automatic re-execution: '+str(job))
            # Preflight precedes reservation; no CUDA leaves the request prepared.
            self._child('check',job)
            if not read(job/'preflight.json')['cuda_available']:raise RuntimeError('CUDA unavailable; task remains prepared: '+str(job))
            gpu_lock=ROOT/'outputs/gpu_test_service/worker.lock';gpu_lock.parent.mkdir(parents=True,exist_ok=True)
            with gpu_lock.open('a') as gpu:
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise RuntimeError('GPU service lock is occupied; task remains prepared')
                # Existing ad hoc jobs may not honor the service lock. Refuse occupied GPUs.
                probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                if probe.stdout.strip():raise RuntimeError('GPU compute processes detected; task remains prepared')
                if (ROOT/'outputs/gpu_test_service/STOP').exists():raise RuntimeError('GPU service STOP is present; task remains prepared')
                write(job/'state.json',dict(state='running',controls_started=None))
                try:self._child(stage,job)
                finally:
                    if (job/'task_result.json').exists():
                        result=read(job/'task_result.json');write(job/'state.json',dict(state=result['status'],result=asset(job/'task_result.json')))
                    else:write(job/'state.json',dict(state='recovery_required',controls_started=None))
            return dict(result=read(job/'task_result.json'),replayed=False,job=str(job))
