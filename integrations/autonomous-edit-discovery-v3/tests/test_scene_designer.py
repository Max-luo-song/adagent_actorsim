
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

import copy,json,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from common_flow import AGENT,write,read,asset
from scene_designer import Designer,validate,signature,compile_entries,PROGRAM,PARAMETERS
from core import AgentError,Deferred,BudgetExceeded,digest
from provider import FixtureProvider


def context():return dict(known_parameter_signatures=[],global_default={'version':'v15'},catalog=[{'scene':551,'editing_ready':True}])
def value(c):
 return dict(schema='scene_action_proposals_v2',context_sha256=digest(c),decision='propose',note='Test only, not an actual discovered failure',edits=[dict(scene=551,program=PROGRAM,parameters=dict(phases=[dict(action='lane_change',duration_s=3.2,target_speed_ratio=1.,lane='ego_lane'),dict(action='decelerate',duration_s=3.,target_speed_ratio=.3,lane='keep'),dict(action='accelerate',duration_s=3.,target_speed_ratio=1.,lane='keep')]),hypothesis='A new merge/braking combination may expose delayed planning',expected_legal_response='Maintain a legal safe corridor or yield as appropriate')])
def config():return dict(api={},budgets=dict(max_api_requests=10,max_api_tokens=2000000,wall_seconds=100))
def fixture(messages,n):return value(_hugsim_expand(json.loads(messages[-1]['content']))['scene_context'])

class ModelEditTests(unittest.TestCase):
 def test_registered_edit_is_validated_without_inventing_failure(self):
  c=context();self.assertEqual(len(validate(value(c),digest(c),[])),1)
 def test_scene_and_actor_cannot_escape_registered_backend(self):
  for key,v in [('scene',552),('program','arbitrary_insert')]:
   c=context();p=value(c);p['edits'][0][key]=v
   with self.assertRaises(AgentError):validate(p,digest(c),[])
 def test_no_code_or_path_fields(self):
  c=context();p=value(c);p['edits'][0]['script']='print(1)'
  with self.assertRaises(AgentError):validate(p,digest(c),[])
 def test_nan_boolean_and_range_rejected(self):
  for v in (float('nan'),True,7.,3.401):
   c=context();p=value(c);p['edits'][0]['parameters']['phases'][0]['duration_s']=v
   with self.assertRaises(AgentError):validate(p,digest(c),[])
 def test_duplicate_numeric_spelling_does_not_spend_gpu(self):
  c=context();p=value(c);params=p['edits'][0]['parameters'];known=[signature(params)];params['phases'][0]['duration_s']=3.2
  with self.assertRaises(AgentError):validate(p,digest(c),known)
 def test_response_must_bind_current_context(self):
  c=context();p=value(c);p['context_sha256']='bad'
  with self.assertRaises(AgentError):validate(p,digest(c),[])
 def test_model_can_wait_without_fabricated_case(self):
  c=context();p=value(c);p.update(decision='wait',edits=[]);self.assertEqual(validate(p,digest(c),[]),[])
 def test_api_missing_never_sends_or_reserves(self):
  class Missing(FixtureProvider):
   def readiness(self):return ['PATCH_MODEL']
  with tempfile.TemporaryDirectory() as d:
   provider=Missing(fixture);model=Designer(d,config(),{'api_requests':1,'charged_tokens':34660},provider)
   with self.assertRaises(Deferred):model.propose(context())
   self.assertEqual(provider.calls,0);self.assertEqual(model.usage()['charged_tokens'],0)
 def test_real_saved_proposal_and_usage_are_idempotent(self):
  with tempfile.TemporaryDirectory() as d:
   p=FixtureProvider(fixture);model=Designer(d,config(),{'api_requests':1,'charged_tokens':34660},p);first=model.propose(context());again=model.propose(context())
   self.assertEqual(first,again);self.assertEqual(p.calls,1);self.assertEqual(model.usage()['api_requests'],1);self.assertEqual(model.usage()['reported_tokens'],32)
   self.assertFalse(first['real_model_response'])
   with self.assertRaises(AgentError):compile_entries(first,1)
 def test_invalid_model_edit_gets_feedback_before_retry(self):
  def reply(messages,n):
   payload=_hugsim_expand(json.loads(messages[-1]['content']));r=value(payload['scene_context'])
   if n==1:r['edits'][0]['parameters']['phases'][0]['duration_s']=True
   else:self.assertTrue(payload['validation_feedback'])
   return r
  with tempfile.TemporaryDirectory() as d:
   p=FixtureProvider(reply);model=Designer(d,config(),{'api_requests':1,'charged_tokens':34660},p);model.propose(context());self.assertEqual(p.calls,2);self.assertEqual(model.usage()['charged_tokens'],64)
 def test_design_and_patch_share_remaining_token_cap(self):
  with tempfile.TemporaryDirectory() as d:
   p=FixtureProvider(fixture);c=config();c['budgets']['max_api_tokens']=34680;model=Designer(d,c,{'api_requests':1,'charged_tokens':34660},p)
   with self.assertRaises(BudgetExceeded):model.propose(context())
   self.assertEqual(p.calls,0);self.assertEqual(model.usage()['api_requests'],0)
 def test_received_response_is_recovered_without_second_call(self):
  with tempfile.TemporaryDirectory() as d:
   p=FixtureProvider(fixture);model=Designer(d,config(),{},p);model.propose(context());(Path(d)/'batch_001/proposal.json').unlink();model.propose(context());self.assertEqual(p.calls,1);self.assertEqual(model.usage()['reported_tokens'],32)
 def test_unknown_interrupted_request_is_not_replayed(self):
  with tempfile.TemporaryDirectory() as d:
   p=FixtureProvider(fixture);model=Designer(d,config(),{},p);record=Path(d)/'batch_001/request_001.json';model.charge(32,record);model.propose(context())
   self.assertEqual(p.calls,1);self.assertEqual(model.usage()['api_requests'],2);self.assertTrue(record.exists());self.assertEqual(read(record)['status'],'interrupted_unknown_usage')
 def test_compiler_uses_host_id_and_checks_model_receipt(self):
  # Synthetic receipt tests host compilation only; no API/model/GPU is executed.
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=context();v=value(c);write(root/'context.json',c);write(root/'response.json',dict(metadata=dict(external_API_called=True,finish_reason='stop'),content=v))
   p=dict(real_model_response=True,proposal=v,context=asset(root/'context.json'),response=asset(root/'response.json'))
   entries=compile_entries(p,1);self.assertTrue(entries[0]['id'].startswith('llm_s551_b001_'));self.assertEqual(set(entries[0]),{'id','phases'})
   write(root/'response.json',dict(metadata=dict(external_API_called=True,finish_reason='stop'),content={}))
   with self.assertRaises(Exception):compile_entries(p,1)
 def test_no_manual_grid_fallback_when_model_unavailable(self):
  import flow,migration,context_builder,scene_designer
  class Missing:
   def __init__(self,*a,**k):pass
   def propose(self,*a,**k):raise Deferred('No API configured')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'global.json',{'version':'v15'});write(root/'source_state.json',{'usage':{}});write(root/'source_config.json',config());write(root/'ledger.json',{'gpu_attempts':0})
   m=dict(host_pins={},global_patch=asset(root/'global.json'),explicit_budget_extension={'gpu_attempts':16},scenario_design=dict(max_batches=1,max_requests_per_batch=3))
   prior=tempfile.TemporaryFile()
   with patch.object(flow,'OUT',root),patch.object(migration,'acquire_previous',return_value=prior),patch.object(migration,'settle_pending'),patch.object(flow,'initialize',return_value=m),patch.object(flow,'remaining',return_value=100),patch.object(scene_designer,'Designer',Missing),patch.object(context_builder,'build',return_value=context()),patch.object(flow,'probe') as gpu:
    with self.assertRaises(Deferred):flow.run()
    gpu.assert_not_called();self.assertFalse((root/'tasks').exists())

if __name__=='__main__':unittest.main()

class AutomaticHandoffTests(unittest.TestCase):
 def test_model_proposal_drives_trial_repeat_and_patch_handoff(self):
  import flow,migration,context_builder,scene_designer
  def metric(score=.27):return dict(status='completed',controls=188,observations=189,collision=False,map_departure=False,unresolved_stop=False,score=dict(hdscore=score,pdms=1.,rc=score),termination_reason='source_scene_time_window_complete',error_stage=None)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=context();v=value(c)
   write(root/'global.json',{'version':'v15'});write(root/'ledger.json',{'gpu_attempts':0});write(root/'source_state.json',{'usage':{}});write(root/'source_config.json',config())
   original=metric(.276)
   for k in ('report','summary','contract'):write(root/(k+'.json'),{});original[k]=asset(root/(k+'.json'))
   write(root/'global_v15.json',{'version':'v15','metrics':{'original551':original}})
   m=dict(host_pins={},global_patch=asset(root/'global.json'),policies={'global':str(root/'global_v15.json')},explicit_budget_extension={'gpu_attempts':16},scenario_design=dict(max_batches=1,max_requests_per_batch=3),new_edits=[])
   class Model:
    def __init__(self,folder,*args,**kwargs):self.folder=Path(folder)
    def propose(self,*args):
     b=self.folder/'batch_001';write(b/'context.json',c);write(b/'response.json',dict(metadata={'external_API_called':True,'finish_reason':'stop'},content=v))
     p=dict(real_model_response=True,proposal=v,context=asset(b/'context.json'),response=asset(b/'response.json'),author={'test_fixture_only':True});write(b/'proposal.json',p);return p
   def compile_audit(args,**kwargs):
    name=args[args.index('--edit')+1];folder=root/'tasks'/name;write(folder/'task.json',{'id':'new_'+name});write(folder/'feasibility.json',dict(passed=True))
   failure=metric(.18);failure.update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action')
   prior=tempfile.TemporaryFile()
   with patch.object(flow,'OUT',root),patch.object(migration,'acquire_previous',return_value=prior),patch.object(migration,'settle_pending'),patch.object(flow,'initialize',return_value=m),patch.object(flow,'remaining',return_value=100),patch.object(scene_designer,'Designer',Model),patch.object(context_builder,'build',return_value=c),patch.object(flow.subprocess,'run',side_effect=compile_audit),patch.object(flow,'probe',side_effect=[failure,copy.deepcopy(failure)]) as trials,patch.object(flow,'same_world',return_value=True),patch.object(flow,'admit') as admission,patch.object(flow,'handoff') as handoff:
    flow.run()
    self.assertEqual(trials.call_count,2);self.assertTrue(all(call.args[2]==m['policies']['global'] for call in trials.call_args_list));admission.assert_called_once();handoff.assert_called_once()
    self.assertTrue(m['new_edits'][0]['id'].startswith('llm_s551_'));self.assertEqual(set(m['new_edits'][0]),{'id',*PARAMETERS})


class StrictRecoveryTests(unittest.TestCase):
 def test_duplicate_key_cached_json_is_not_silently_normalized(self):
  with tempfile.TemporaryDirectory() as d:
   provider=FixtureProvider(fixture);designer=Designer(d,config(),{},provider);record=Path(d)/'batch_001/request_001.json';designer.charge(32,record)
   c=context();raw=json.dumps(value(c));raw=raw.replace('"decision": "propose"','"decision": "wait", "decision": "propose"')
   write(record,dict(metadata=dict(usage_tokens=32,reserved_tokens=32,usage_known=True,finish_reason='stop'),content=raw))
   designer.propose(c);self.assertEqual(provider.calls,1);self.assertEqual(designer.usage()['api_requests'],2)
