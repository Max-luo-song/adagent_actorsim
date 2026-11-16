
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

import json,tempfile,unittest,copy
pass  # Path is provided by the portable boundary above
from helpers import config,FakeAdapter,proposal,reply,LOCAL
from core import AgentError,read,write,asset
from contracts import require_keys,decision
from evidence import diagnostics,restore_diagnostics,decorate
from engine import Engine
from provider import FixtureProvider
class WorkflowRepairTests(unittest.TestCase):
 def test_format_retry_receives_previous_draft_not_only_error_message(self):
  def fixture(messages,n):
   data=_hugsim_expand(json.loads(messages[-1]['content']));p=proposal(data['evidence'])
   if n==1:p.pop('diagnostics')
   else:
    prior=data['previous_response'];prior=_hugsim_expand(json.loads(prior)) if isinstance(prior,str) else prior
    self.assertEqual(prior['patch_instruction'],p['patch_instruction']);self.assertNotIn('diagnostics',prior);self.assertIn('diagnostics',data['validation_feedback']['message'])
   return p
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(fixture),FakeAdapter(root,[.262]));e.initialize('correction');s=e.run();self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['usage']['api_requests'],2)

 def test_many_cached_queries_fit_without_losing_score_facts(self):
  from evidence import bounded
  e={'items':{'source':{'text':'x'*1000},'report_1':{'facts':{'score':.3,'sampled_frames':[{'large':'x'*900}]}},'diagnostic_2':{'facts':[{'index':0,'state':'x'*100}]}}}
  result=bounded(e,1600);self.assertEqual(result['items']['report_1']['facts']['score'],.3);self.assertTrue(result['items']['report_1']['facts']['sampled_frames_omitted_for_budget']);self.assertIn('diagnostic_2',result['items']);self.assertEqual(result['items']['source']['text'],'x'*1000)
 def test_latest_overview_is_automatic_and_tracks_round(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e=Engine(root/'campaign',config(),FixtureProvider(reply),FakeAdapter(root));s=e.initialize('overview');r=e.new_round(s);text=(root/'campaign/LATEST.md').read_text();self.assertIn('overview_r001',text);self.assertIn('尚未生成',text)

 def test_missing_field_feedback_names_actual_field(self):
  with self.assertRaisesRegex(AgentError,"Missing fields:.*diagnostics"):
   require_keys({'schema':'x'},('schema','diagnostics'))
  with self.assertRaisesRegex(AgentError,'Expected one JSON object; got list'):require_keys([{}],('schema',))
 def test_finish_duplicate_feedback_does_not_crash_and_preserves_result(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e=Engine(root/'campaign',config(),FixtureProvider(reply),FakeAdapter(root));s=e.initialize('duplicate');r=e.new_round(s);r['feedback']={'type':'specific failure'}
   e.finish(s,r,'rejected','invalid_model_decision',feedback=r['feedback']);self.assertIsNone(s['active_round']);self.assertEqual(read(root/'campaign/rounds/duplicate_r001/round_result.json')['feedback'],r['feedback'])
 def test_six_invalid_models_end_normally_without_executor_error(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;p=FixtureProvider(lambda m,n:[{}]);e=Engine(root/'campaign',c,p,FakeAdapter(root));e.initialize('invalid');s=e.run();self.assertEqual(p.calls,6);self.assertEqual(s['history'][0]['reason'],'invalid_model_decision');self.assertEqual(s['status'],'budget_exhausted')
 def test_rejected_diagnostic_does_not_consume_successful_quota(self):
  def fixture(messages,n):
   e=_hugsim_expand(json.loads(messages[-1]['content']))['evidence'];p=proposal(e,'diagnose' if n==1 else 'patch')
   if n==1:p['diagnostics']=[dict(kind='trace_window',item_id='source',start=0,end=1)]
   else:self.assertEqual(e['diagnostic_runtime']['remaining'],2);self.assertEqual(e['diagnostic_runtime']['rejected_requests'],1)
   return p
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(fixture),FakeAdapter(root,[.262]));e.initialize('quota');s=e.run();self.assertEqual(s['history'][0]['status'],'accepted')
 def test_real_v4_windows_now_fit_and_retain_source_hash(self):
  base=LOCAL.parents[1]/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/autonomous_patch_agent_v4/campaigns/patch_auto_v4/rounds'
  count=0
  for folder in sorted(base.iterdir()):
   evidence=read(folder/'evidence_capture_000.json')
   for p in sorted(folder.glob('decision_*.json')):
    requests=read(p)['decision'].get('diagnostics',[])
    if not requests:continue
    result,ids=diagnostics(evidence,requests,80000);self.assertLessEqual(len(json.dumps(result,ensure_ascii=False)),80000);self.assertEqual(result['items']['source'],evidence['items']['source']);self.assertEqual(len(ids),len(requests))
    for key in ids:
     item=result['items'][key];self.assertIn('source_asset',item)
     if 'sampling' in item:self.assertTrue(item['sampling']['full_report_retained']);self.assertGreater(len(item['facts']),0)
    count+=1
  self.assertGreaterEqual(count,6)
 def test_diagnostic_cache_relinks_changed_report_ids_and_deduplicates(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'run_report.json';write(path,{'trace':[dict(index=0,motion_state=[0,0,0,1,0],road_reference={},road_gate={})]});e={'items':{'r':{'asset':asset(path)}}};req=dict(kind='trace_window',item_id='r',start=0,end=0);new,ids=diagnostics(e,[req],80000);again,ids2=diagnostics(new,[req],80000);self.assertEqual(ids,ids2);self.assertEqual(new,again)
   restored=restore_diagnostics({'items':{'renamed':{'asset':asset(path)}}},{ids[0]:new['items'][ids[0]]},80000);self.assertEqual(restored['items'][ids[0]]['source_item'],'renamed')
 def test_diagnostics_persist_into_later_round(self):
  def fixture(messages,n):
   e=_hugsim_expand(json.loads(messages[-1]['content']))['evidence'];p=proposal(e,'diagnose' if n==1 else 'patch')
   if n==1:p['diagnostics']=[dict(kind='source_method',item_id='source',start=0,end=2)]
   elif n==3:self.assertTrue(any(k.startswith('diagnostic_') for k in e['items']))
   return p
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e=Engine(root/'campaign',config(),FixtureProvider(fixture),FakeAdapter(root,[.259,.262]));e.initialize('carry');s=e.run();self.assertTrue(s['diagnostic_cache']);self.assertEqual(len(s['history']),2)
if __name__=='__main__':unittest.main()
