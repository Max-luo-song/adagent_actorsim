"""Owned resident UniAD service with isolated, synchronous driving episodes."""

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

import json
import os
pass  # Path is provided by the portable boundary above
import pickle
import shutil
import subprocess
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'code/UniAD_SIM').is_dir())


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


class ResidentInferenceService:
    def __init__(self,folder,launch=None,startup_timeout=1200,response_timeout=300):
        self.folder=Path(folder).resolve();self.folder.mkdir(parents=True,exist_ok=False)
        self.job=self.folder/'worker';self.job.mkdir()
        self.launch=launch or subprocess.Popen
        self.startup_timeout=startup_timeout;self.response_timeout=response_timeout
        self.process=None;self.log=None;self.ready=None;self.active=None
        self.next_index=0;self.tokens=set();self.records=[];self.episodes=[]
        self.closed=False;self.poisoned=False

    def _wait(self,path,timeout):
        deadline=time.monotonic()+timeout;progress=time.monotonic()+15
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'Resident UniAD exited ({self.process.returncode}); see {self.folder/"worker.log"}')
            if time.monotonic()>=deadline:
                raise TimeoutError(f'Resident UniAD waiting for {path.name}; see {self.folder/"worker.log"}')
            if time.monotonic()>=progress:
                print(f'Resident driving: waiting for {path.name}; log: {self.folder/"worker.log"}',flush=True)
                progress+=15
            time.sleep(.05)
        return _hugsim_expand(json.loads(path.read_text()))

    def _start(self,initial_packet):
        first=self.job/'step_000';first.mkdir()
        shutil.copyfile(initial_packet,first/'observation.pkl')
        env=dict(os.environ,BASE=str(ROOT),PYTHONUNBUFFERED='1',BEV_EXPERIMENT_SEED='20261021',
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',UNIAD_INIT_TRACE='1',
            UNIAD_STARTUP_REPORT=str(self.job/'startup.json'))
        for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
        cmd=[str(ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'),
            str(ROOT/'waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py'),
            '--job',str(self.job),'--planning-bev','current','--model-profile','calibration_geometry_v1',
            '--diagnostic-closed-loop','--session-lifecycle']
        self.log=(self.folder/'worker.log').open('x')
        self.process=self.launch(cmd,cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT)
        self.ready=self._wait(self.job/'ready.json',self.startup_timeout)
        if (not self.ready.get('session_lifecycle') or self.ready.get('uniad_model_loads')!=1
                or self.ready.get('pid')!=self.process.pid or not self.ready.get('resident_identity')):
            raise RuntimeError('Resident worker did not expose the verified lifecycle protocol')
        atomic_json(self.folder/'ready.json',self.ready)

    def begin_episode(self,worker_folder,initial_packet,token):
        if self.closed or self.poisoned:raise RuntimeError('Resident service is closed or failed')
        if self.active is not None:raise RuntimeError('End the current episode before starting another')
        if not isinstance(token,str) or not token or token in self.tokens:
            raise ValueError('Resident episodes require a new nonempty scene token')
        with Path(initial_packet).open('rb') as stream:_,info=pickle.load(stream)
        if info['scene_token']!=token:raise ValueError('Initial packet token differs from episode')
        try:
            if self.process is None:self._start(initial_packet)
            elif self.process.poll() is not None:raise RuntimeError('Resident worker is no longer running')
            folder=Path(worker_folder).resolve()
            self.active=dict(token=token,worker_folder=str(folder),start_service_index=self.next_index,
                local_count=0,episode_number=len(self.episodes)+1)
            self.tokens.add(token)
            atomic_json(folder/'ready.json',self.ready)
            return dict(self.ready,resident_reused=len(self.episodes)>0,episode_number=self.active['episode_number'])
        except BaseException:
            self.poisoned=True
            raise

    def infer(self,packet_path,index,seed=None):
        if self.closed or self.poisoned or self.active is None:raise RuntimeError('No healthy resident episode')
        if index!=self.active['local_count']:raise ValueError('Non-contiguous episode inference index')
        with Path(packet_path).open('rb') as stream:_,info=pickle.load(stream)
        if info['scene_token']!=self.active['token']:raise ValueError('Packet belongs to a different episode')
        if (index==0)!=(seed is not None):raise ValueError('Seed is required only on the first episode frame')
        global_index=self.next_index
        req=dict(index=global_index,packet=str(Path(packet_path).resolve()))
        if seed is not None:req['diagnostic_seed']=seed
        try:
            atomic_json(self.job/f'request_{global_index:03d}.json',req)
            source=self._wait(self.job/f'done_{global_index:03d}.json',self.response_timeout)
            if source.get('status')!='ok':raise RuntimeError(f'Resident inference failed: {source}')
            remote=self.job/f'step_{global_index:03d}'
            state=_hugsim_expand(json.loads((remote/'session_lifecycle.json').read_text()))
            temporal=_hugsim_expand(json.loads((remote/'temporal_state.json').read_text()))
            if state['identity']!=self.ready['resident_identity']:
                raise RuntimeError('Resident model identity changed between driving steps')
            if (state['reset']!=(index==0) or temporal['history_bev_present']!=(index>0)
                    or state['reset_count']!=self.active['episode_number']
                    or (index==0 and not state['reset_state_cleared'])):
                raise RuntimeError('Resident driving episode history was not isolated')
            local_root=Path(self.active['worker_folder']);local=local_root/f'step_{index:03d}'
            local.mkdir(exist_ok=True)
            for path in remote.iterdir():
                if not path.is_file() or path.name=='observation.pkl':continue
                target=local/path.name
                if target.exists():raise FileExistsError(target)
                try:os.link(path,target)
                except OSError:shutil.copyfile(path,target)
            response=dict(source,index=index,service_index=global_index,resident_pid=self.process.pid)
            for key in ('perception','reliability'):
                response[key]=str(local/Path(source[key]).name)
            atomic_json(local_root/f'done_{index:03d}.json',response)
            atomic_json(local_root/f'resident_source_done_{index:03d}.json',source)
            self.records.append(dict(service_index=global_index,episode_token=self.active['token'],
                local_index=index,packet=str(packet_path),identity=state['identity'],
                reset=state['reset'],history=temporal['history_bev_present']))
            self.active['local_count']+=1;self.next_index+=1
            atomic_json(self.folder/'inferences.json',self.records)
            return response
        except BaseException as exc:
            self.poisoned=True
            atomic_json(self.folder/'failure.json',dict(error=repr(exc),service_index=global_index))
            raise

    def end_episode(self,completed):
        if self.active is None:return
        episode=dict(self.active,completed=bool(completed),worker_pid=self.process.pid)
        self.episodes.append(episode)
        atomic_json(Path(episode['worker_folder'])/'resident_episode.json',episode)
        atomic_json(self.folder/'episodes.json',self.episodes)
        self.active=None

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.process is not None:
                atomic_json(self.job/'stop.json',{})
                try:self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        finally:
            if self.log is not None:self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
