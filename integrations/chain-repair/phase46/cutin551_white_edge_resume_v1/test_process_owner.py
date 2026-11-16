
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

import json,subprocess,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from resume_store import atomic_json
from process_owner import record,identity,cleanup
class ProcessOwnerTests(unittest.TestCase):
 def fixture(self,root):
  root=Path(root);worker=root/'fake_worker.py';worker.write_text('import time\ntime.sleep(60)\n');job=root/'attempt_001/uniad_worker';job.mkdir(parents=True);p=subprocess.Popen([sys.executable,str(worker),'--job',str(job)]);record(job,p.pid);return worker,job,p
 def test_only_exact_owned_worker_is_stopped(self):
  with tempfile.TemporaryDirectory() as tmp:
   worker,job,p=self.fixture(tmp)
   try:self.assertEqual(cleanup(tmp,worker,sys.executable),1);p.wait(timeout=5)
   finally:
    if p.poll() is None:p.terminate();p.wait()
 def test_pid_start_time_mismatch_never_signals_process(self):
  with tempfile.TemporaryDirectory() as tmp:
   worker,job,p=self.fixture(tmp)
   try:
    file=job/'owned_process.json';d=_hugsim_expand(json.loads(file.read_text()));d['start_ticks']=str(int(d['start_ticks'])+1);atomic_json(file,d);self.assertEqual(cleanup(tmp,worker,sys.executable),0);self.assertIsNone(p.poll())
   finally:p.terminate();p.wait()
 def test_worker_command_mismatch_is_not_killed(self):
  with tempfile.TemporaryDirectory() as tmp:
   worker,job,p=self.fixture(tmp)
   try:
    with self.assertRaisesRegex(RuntimeError,'command mismatch'):cleanup(tmp,worker.with_name('other.py'),sys.executable)
    self.assertIsNone(p.poll())
   finally:p.terminate();p.wait()
if __name__=='__main__':unittest.main(verbosity=2)
