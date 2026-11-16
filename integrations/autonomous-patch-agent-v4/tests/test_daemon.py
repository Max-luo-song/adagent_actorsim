
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
from helpers import metrics
from core import process_identity,write
import daemon
class DaemonTests(unittest.TestCase):
 def test_coordinator_detach_is_idempotent_and_does_not_persist_key(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);command=[sys.executable,'-c','import time;time.sleep(.4)'];env=dict(os.environ,PATCH_API_KEY='test_fixture_secret')
   first=daemon.launch(p,command,p,env);second=daemon.launch(p,command,p,env)
   self.assertEqual(first['pid'],second['pid']);self.assertEqual(second['status'],'already_running');self.assertNotIn('test_fixture_secret',(p/'coordinator_owner.json').read_text());self.assertTrue(daemon.active(p))
   for process in daemon._HANDLES:process.wait(timeout=5)
   self.assertIsNone(daemon.active(p))
 def test_status_exposes_live_frame_without_dumping_full_reports(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);out=p/'matrix';write(out/'cases/cutin551/on/B/progress.json',{'index':43,'total_controls':188,'stage':'live_light_tracking'})
   state=dict(champion=dict(version='fixture',metrics=metrics()),reference_metrics={},history=[],active_round=dict(id='r001',phase='validate',current_job='job',attempt={'output':str(out)}),status='running_gpu')
   s=daemon.summary(state,p);self.assertEqual(s['episode_progress']['cutin551']['index'],43);self.assertNotIn('active_round',s);self.assertEqual(s['champion_version'],'fixture')
if __name__=='__main__':unittest.main()
