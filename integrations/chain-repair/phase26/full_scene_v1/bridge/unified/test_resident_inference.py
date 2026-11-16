"""Real pipe/file protocol with a CPU fake worker; no model claims."""

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
pass  # Path is provided by the portable boundary above
import pickle
import subprocess
import sys
import tempfile
import unittest

from resident_inference import ResidentInferenceService


STUB='''import json,os,pickle,sys,time
from pathlib import Path
import numpy as np
job=Path(sys.argv[1]);leak=int(sys.argv[2]);previous=None;counter=0;i=0
identity=dict(pid=os.getpid(),model_object_id=11,first_parameter_object_id=12,first_parameter_data_ptr=13,model_loads=1)
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
write(job/'ready.json',dict(session_lifecycle=True,uniad_model_loads=1,pid=os.getpid(),resident_identity=identity,planning_bev='current'))
while not (job/'stop.json').exists():
 req=job/f'request_{i:03d}.json'
 if not req.exists():time.sleep(.01);continue
 value=json.loads(req.read_text())
 with Path(value['packet']).open('rb') as f:_,info=pickle.load(f)
 new=info['scene_token']!=previous
 if new:counter+=1
 assert ('diagnostic_seed' in value)==new
 out=job/f'step_{i:03d}';out.mkdir(exist_ok=True)
 history=not new or (leak and counter==2)
 write(out/'temporal_state.json',dict(history_bev_present=bool(history)))
 write(out/'session_lifecycle.json',dict(identity=identity,reset=new,reset_state_cleared=new,reset_count=counter))
 np.savez(out/'perception.npz',trajectory=np.zeros((6,2)))
 write(out/'scene_reliability.json',dict(fake_worker=True))
 write(job/f'done_{i:03d}.json',dict(status='ok',index=i,timestamp=info['timestamp'],perception=str(out/'perception.npz'),reliability=str(out/'scene_reliability.json')))
 previous=info['scene_token'];i+=1
'''


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.stub=self.root/'stub.py';self.stub.write_text(STUB)
        self.launch_count=0;self.leak=False
        def launch(command,**kwargs):
            self.launch_count+=1
            job=command[command.index('--job')+1]
            return subprocess.Popen([sys.executable,'-u',str(self.stub),job,str(int(self.leak))],**kwargs)
        self.service=ResidentInferenceService(self.root/'service',launch=launch,startup_timeout=5,response_timeout=5)
        self.addCleanup(self.service.close)

    def packet(self,episode,index,token):
        folder=self.root/episode/f'step_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'observation.pkl'
        with path.open('wb') as stream:pickle.dump(({'fake':True},dict(scene_token=token,timestamp=8.+index*.1)),stream,protocol=4)
        return path

    def test_two_driving_episodes_use_one_worker_and_remap_step_indices(self):
        pid=None
        for episode,token in [('first','A'),('second','B')]:
            first=self.packet(episode,0,token)
            ready=self.service.begin_episode(first.parent.parent,first,token)
            if pid is None:pid=ready['pid']
            self.assertEqual(ready['pid'],pid)
            for i in (0,1):
                packet=self.packet(episode,i,token)
                result=self.service.infer(packet,i,seed=20261021 if i==0 else None)
                self.assertEqual(result['index'],i)
                self.assertTrue(Path(result['perception']).is_file())
                self.assertTrue((packet.parent/'session_lifecycle.json').is_file())
                expected=i+(2 if episode=='second' else 0)
                self.assertEqual(result['service_index'],expected)
            self.service.end_episode(completed=True)
            self.assertIsNone(self.service.process.poll())  # episode end did not kill the model
            self.assertFalse((self.service.job/'stop.json').exists())
        self.assertEqual(self.launch_count,1)
        self.assertEqual(len(self.service.records),4)
        self.service.close()
        self.assertIsNotNone(self.service.process.returncode)

    def test_stale_token_and_parallel_episode_are_rejected(self):
        first=self.packet('first',0,'A')
        self.service.begin_episode(first.parent.parent,first,'A')
        with self.assertRaises(RuntimeError):self.service.begin_episode(first.parent.parent,first,'B')
        bad=self.packet('first',1,'stale')
        with self.assertRaises(ValueError):self.service.infer(bad,0,seed=20261021)
        self.assertEqual(self.service.next_index,0)
        self.service.end_episode(False)
        with self.assertRaises(ValueError):self.service.begin_episode(first.parent.parent,first,'A')

    def test_leaked_first_frame_history_poisoned_service_and_retains_failure(self):
        self.leak=True
        for ep,token in [('first','A'),('second','B')]:
            packet=self.packet(ep,0,token)
            self.service.begin_episode(packet.parent.parent,packet,token)
            if token=='A':
                self.service.infer(packet,0,seed=20261021);self.service.end_episode(True)
            else:
                with self.assertRaisesRegex(RuntimeError,'history was not isolated'):
                    self.service.infer(packet,0,seed=20261021)
        self.assertTrue(self.service.poisoned)
        self.assertTrue((self.service.folder/'failure.json').is_file())

    def test_wrong_initial_packet_never_starts_a_worker(self):
        packet=self.packet('first',0,'other')
        with self.assertRaises(ValueError):self.service.begin_episode(packet.parent.parent,packet,'wanted')
        self.assertEqual(self.launch_count,0)


if __name__=='__main__':unittest.main()
