
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

import copy,json,sys,tempfile,time,unittest
pass  # Path is provided by the portable boundary above
from helpers import LOCAL,FakeAdapter,config,reply,proposal
from core import AgentError,Deferred,asset,lock,read,write
from engine import Engine
from provider import FixtureProvider
class EngineTests(unittest.TestCase):
 def test_accept_then_reject_keeps_prior_champion(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);provider=FixtureProvider(reply);adapter=FakeAdapter(root,[.262,.261]);engine=Engine(root/'campaign',config(),provider,adapter);engine.initialize('fixture');state=engine.run()
   self.assertEqual([r['status'] for r in state['history']],['accepted','rejected']);self.assertEqual(state['champion']['version'],'fixture_r001');self.assertTrue(state['simulation_only']);self.assertFalse(state['global_patch_validated']);self.assertFalse(state['production_driver_changed']);self.assertEqual(provider.calls,2)
   first=read(root/'campaign/rounds/fixture_r001/acceptance.json');self.assertFalse(first['GPU_executed']);self.assertTrue(first['simulation_only']);self.assertEqual(read(root/'campaign/working_champion.json')['version'],'fixture_r001')
 def test_invalid_model_json_is_repaired_before_application(self):
  def fixture(messages,n):
   p=reply(messages,n)
   if n==1:p['edit_instruction']['cases']=['not_registered']
   return p
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;provider=FixtureProvider(fixture);e=Engine(root/'campaign',c,provider,FakeAdapter(root,[.262]));e.initialize('repair');s=e.run()
   self.assertEqual(provider.calls,2);self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['usage']['api_requests'],2)
 def test_diagnostics_are_bound_to_new_evidence_digest(self):
  def fixture(messages,n):
   evidence=_hugsim_expand(json.loads(messages[-1]['content']))['evidence'];p=proposal(evidence,'diagnose' if n==1 else 'patch')
   if n==1:p['diagnostics']=[dict(kind='source_method',item_id='source',start=0,end=2)]
   else:self.assertIn('diagnostic_1',evidence['items'])
   return p
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(fixture),FakeAdapter(root,[.262]));e.initialize('diagnose');s=e.run();self.assertEqual(s['history'][0]['status'],'accepted');self.assertTrue((root/'campaign/rounds/diagnose_r001/evidence_01.json').exists())
 def test_no_gpu_after_offline_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),FakeAdapter(root,fail_cpu=True));e.initialize('offline');s=e.run()
   self.assertEqual(s['usage']['gpu_dispatches'],0);self.assertEqual(s['champion']['version'],'fixture_seed');self.assertEqual(s['history'][0]['reason'],'offline_validation_failed')
 def test_waiting_for_later_api_does_not_spend_wall_budget(self):
  class Missing(FixtureProvider):
   def readiness(self):return ['PATCH_MODEL']
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['wall_seconds']=.01;e=Engine(root/'campaign',c,Missing(reply),FakeAdapter(root));e.initialize('pending');time.sleep(.03);s=e.run()
   self.assertEqual(s['status'],'waiting_api');self.assertIsNone(s['started_epoch']);self.assertEqual(s['usage']['api_requests'],0)
 def test_token_budget_stops_before_model_or_gpu(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_api_tokens']=1;p=FixtureProvider(reply);e=Engine(root/'campaign',c,p,FakeAdapter(root));e.initialize('budget');s=e.run()
   self.assertEqual(s['status'],'budget_exhausted');self.assertEqual(p.calls,0);self.assertEqual(s['usage']['gpu_dispatches'],0)
 def test_gpu_busy_is_deferred_without_rejecting_or_spending_attempt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),FakeAdapter(root,[.262],busy_once=True));e.initialize('busy');s=e.run()
   self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['usage']['gpu_dispatches'],2);self.assertEqual(s['usage']['gpu_attempts'],1)
 def test_fresh_coordinator_observes_live_job_without_redispatch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;p=FixtureProvider(reply);a=FakeAdapter(root,[.262],slow=True);e=Engine(root/'campaign',c,p,a);e.initialize('resume')
   for _ in range(20):
    s=e.run(one_step=True)
    if (s.get('active_round') or {}).get('current_job'):break
   self.assertTrue((s.get('active_round') or {}).get('current_job'))
   replacement=Engine(root/'campaign',provider=p,adapter=a);s=replacement.run();self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(p.calls,1);self.assertEqual((root/'campaign/rounds/resume_r001/dispatch_count').read_text(),'1')
 def test_stop_and_integrity_checks(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=FixtureProvider(reply);e=Engine(root/'campaign',config(),p,FakeAdapter(root));e.initialize('stopper');write(root/'campaign/STOP',{'reason':'test'});s=e.run();self.assertEqual(s['status'],'stopped');self.assertEqual(p.calls,0)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e=Engine(root/'campaign',config(),FixtureProvider(reply),FakeAdapter(root));e.initialize('tamper');c=read(root/'campaign/config.json');c['acceptance']['original_max_hd_drop']=1;write(root/'campaign/config.json',c);s=e.run();self.assertEqual(s['status'],'integrity_or_executor_error')
 def test_model_stop_and_coordinator_lock_do_not_apply_code(self):
  def fixture(messages,n):return proposal(_hugsim_expand(json.loads(messages[-1]['content']))['evidence'],'stop')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e=Engine(root/'campaign',config(),FixtureProvider(fixture),FakeAdapter(root));e.initialize('modelstop')
   with lock(root/'campaign/coordinator.lock'):
    with self.assertRaises(Deferred):e.run()
   s=e.run();self.assertEqual(s['status'],'model_stopped');self.assertEqual(s['usage']['gpu_dispatches'],0)
if __name__=='__main__':unittest.main()

class RecoveryBoundaryTests(unittest.TestCase):
 def test_gpu_timeout_uses_new_attempt_without_rerunning_earlier_cpu_stage(self):
  class Adapter(FakeAdapter):
   def stages(self,rd,b,a,c):
    s=super().stages(rd,b,a,c)
    if a['number']==1:s[-1].update(command=[sys.executable,'-c','import time;time.sleep(3)'],timeout=.2)
    return s
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),Adapter(root,[.262]));e.initialize('gpuretry');s=e.run()
   self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['usage']['gpu_attempts'],2);self.assertEqual((root/'campaign/rounds/gpuretry_r001/dispatch_count').read_text(),'1');self.assertTrue((root/'campaign/rounds/gpuretry_r001/attempt_1').exists());self.assertTrue((root/'campaign/rounds/gpuretry_r001/attempt_2/evaluation.json').exists())
 def test_interrupted_cpu_retries_that_stage_before_gpu(self):
  class Adapter(FakeAdapter):
   def stages(self,rd,b,a,c):
    s=super().stages(rd,b,a,c);marker=Path(rd)/'cpu_marker';script="from pathlib import Path;import time;m=Path("+repr(str(marker))+");first=not m.exists();m.write_text(str(int(m.read_text())+1) if m.exists() else '1');time.sleep(3 if first else 0);"+s[0]['command'][-1];s[0].update(command=[sys.executable,'-c',script],timeout=.3);return s
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),Adapter(root,[.262]));e.initialize('cpuretry');s=e.run()
   self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual((root/'campaign/rounds/cpuretry_r001/cpu_marker').read_text(),'2');self.assertEqual(s['usage']['gpu_attempts'],1)
 def test_transient_api_reservation_is_retained_before_success(self):
  from provider import TransientAPI
  class Temporary(FixtureProvider):
   def complete(self,messages,schema,before,record):
    if self.calls==0:self.calls+=1;before(50);write(record,{'fixture_transient':True});raise TransientAPI('Fixture temporary failure')
    return super().complete(messages,schema,before,record)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,Temporary(reply),FakeAdapter(root,[.262]));e.initialize('apiretry');s=e.run()
   self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['usage']['api_requests'],2);self.assertEqual(s['usage']['charged_tokens'],82)
