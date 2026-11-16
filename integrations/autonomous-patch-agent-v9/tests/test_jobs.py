
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

import os,sys,tempfile,time,unittest
pass  # Path is provided by the portable boundary above
from helpers import LOCAL
from core import AgentError,read,sha,write
import jobs
class JobTests(unittest.TestCase):
 def await_result(self,folder):
  limit=time.time()+12
  while time.time()<limit:
   r=jobs.inspect(folder)
   if r['status']!='running':return r
   time.sleep(.05)
  jobs.cancel(folder);self.fail('Job did not complete')
 def test_success_is_idempotent_and_output_mutation_breaks_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);out=p/'out';count=p/'count';job=p/'job'
   script="from pathlib import Path;p=Path("+repr(str(count))+");p.write_text(str(int(p.read_text())+1) if p.exists() else '1');Path("+repr(str(out))+").write_text('ok')"
   command=[sys.executable,'-B','-c',script];jobs.launch(job,command,p,5,{},[out],p/'STOP');r=self.await_result(job);self.assertEqual(r['status'],'succeeded')
   jobs.launch(job,command,p,5,{},[out],p/'STOP');self.assertEqual(count.read_text(),'1');out.write_text('tamper')
   with self.assertRaises(AgentError):jobs.inspect(job)
 def test_live_job_is_reused_after_coordinator_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);command=[sys.executable,'-B','-c','import time;time.sleep(.4)'];first=jobs.launch(p/'job',command,p,5,{},[],p/'STOP');second=jobs.launch(p/'job',command,p,4,{},[],p/'STOP')
   self.assertEqual(second['status'],'running');self.assertEqual(self.await_result(p/'job')['status'],'succeeded')
 def test_timeout_and_missing_output_do_not_pass(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);jobs.launch(p/'timeout',[sys.executable,'-c','import time;time.sleep(5)'],p,.2,{},[],p/'STOP');self.assertEqual(self.await_result(p/'timeout')['status'],'timeout')
   jobs.launch(p/'missing',[sys.executable,'-c','pass'],p,5,{},[p/'absent'],p/'STOP');self.assertEqual(self.await_result(p/'missing')['status'],'failed')
 def test_source_mutation_during_stage_is_detected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);source=p/'source';source.write_text('before');jobs.launch(p/'job',[sys.executable,'-c','import time;time.sleep(.4)'],p,5,{str(source):sha(source)},[],p/'STOP');source.write_text('after')
   self.assertEqual(self.await_result(p/'job')['status'],'failed')
 def test_stop_cancels_only_registered_job(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);jobs.launch(p/'job',[sys.executable,'-c','import time;time.sleep(5)'],p,8,{},[],p/'STOP');time.sleep(.2);write(p/'STOP',{'reason':'test'});self.assertEqual(self.await_result(p/'job')['status'],'cancelled')
 def test_api_secrets_are_not_forwarded_to_native_jobs(self):
  from unittest.mock import patch
  with patch.dict(os.environ,{'PATCH_API_KEY':'secret','UNRELATED_API_KEY':'another'}):
   e=jobs.environment();self.assertNotIn('PATCH_API_KEY',e);self.assertNotIn('UNRELATED_API_KEY',e)
if __name__=='__main__':unittest.main()

class DispatchBoundaryTests(unittest.TestCase):
 def test_fresh_dispatch_spec_waits_for_metadata_then_becomes_interrupted(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);write(p/'spec.json',{})
   self.assertEqual(jobs.inspect(p)['status'],'running');os.utime(p/'spec.json',(time.time()-10,time.time()-10));self.assertEqual(jobs.inspect(p)['status'],'interrupted')

class ResumeEnvironmentTests(unittest.TestCase):
 def test_existing_job_keeps_original_environment_when_new_terminal_differs(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);command=[sys.executable,'-c','import time;time.sleep(.3)'];jobs.launch(p/'job',command,p,4,{},[],p/'STOP')
   original=read(p/'job/spec.json')['env']
   with patch.dict(os.environ,{'PATH':'/a/different/terminal/path'}):self.assertEqual(jobs.launch(p/'job',command,p,4,{},[],p/'STOP')['status'],'running')
   self.assertEqual(read(p/'job/spec.json')['env'],original)
   limit=time.time()+6
   while jobs.inspect(p/'job')['status']=='running' and time.time()<limit:time.sleep(.05)
   self.assertEqual(jobs.inspect(p/'job')['status'],'succeeded')
