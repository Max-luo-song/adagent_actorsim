"""CPU subprocess protocol tests; the child deliberately does no model inference."""

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

from contextlib import redirect_stdout
import io
import json
pass  # Path is provided by the portable boundary above
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_resident_validation import inference


STUB = '''import json,os,sys,time
from pathlib import Path
import numpy as np
job=Path(sys.argv[1]); drift=float(sys.argv[2])
identity=dict(pid=os.getpid(),model_object_id=42,first_parameter_object_id=43,first_parameter_data_ptr=123,model_loads=1)
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data));temp.replace(path)
write(job/'ready.json',dict(pid=os.getpid(),resident_identity=identity,uniad_model_loads=1))
for i in range(10):
 request=job/f'request_{i:03d}.json'
 while not request.exists():time.sleep(.01)
 req=json.loads(request.read_text()); assert req['index']==i
 assert ('diagnostic_seed' in req)==(i%2==0)
 out=job/f'step_{i:03d}';out.mkdir(exist_ok=True)
 write(out/'temporal_state.json',dict(history_bev_present=bool(i%2)))
 write(out/'session_lifecycle.json',dict(identity=identity,reset=not bool(i%2),reset_state_cleared=True,reset_count=i//2+1))
 data={key:np.full((2,),i%2,dtype=float) for key in ('trajectory','occupancy','candidate_boxes','candidate_scores','candidate_labels')}
 if i>=8:data['candidate_scores']+=drift
 np.savez(out/'perception.npz',**data)
 write(job/f'done_{i:03d}.json',dict(status='ok'))
while not (job/'stop.json').exists():time.sleep(.01)
'''


class ProtocolTests(unittest.TestCase):
    def exercise(self, folder, drift):
        root=Path(folder);stub=root/'stub.py';stub.write_text(STUB)
        episodes=[]
        for i,scene in enumerate([552,552,551,552,552]):
            frames=[]
            for frame in (80,81):
                path=root/f'input_{i}_{frame}.pkl';path.write_bytes(b'CPU protocol fixture; no sensor/model claim')
                frames.append(dict(packet=str(path),weather_audit={'window_frames':[80,80,80]},
                    rgb_sha256={'front':str(scene)}))
            episodes.append(dict(name=str(i),scene=scene,checkpoint=str(scene),frames=frames,
                gaussian_sample_sha256={'Background':str(scene)}))
        launch=subprocess.Popen
        def cpu_child(command,**kwargs):
            return launch([sys.executable,'-u',str(stub),str(root/'uniad_worker'),str(drift)],**kwargs)
        with patch('run_resident_validation.subprocess.Popen',side_effect=cpu_child),redirect_stdout(io.StringIO()):
            inference(root,episodes)
        return _hugsim_expand(json.loads((root/'acceptance.json').read_text()))

    def test_all_ten_requests_use_one_child_and_replay_is_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.exercise(folder,0)
            self.assertTrue(report['passed'])
            self.assertFalse(report['closed_loop_driving'])
            self.assertTrue(report['checks']['one_worker'])
            self.assertTrue(report['checks']['identical_input_outputs_match'])

    def test_replay_drift_is_preserved_as_failed_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'Resident validation failed'):
                self.exercise(folder,.01)
            report=_hugsim_expand(json.loads((Path(folder)/'acceptance.json').read_text()))
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['identical_input_outputs_match'])


if __name__=='__main__':unittest.main()
