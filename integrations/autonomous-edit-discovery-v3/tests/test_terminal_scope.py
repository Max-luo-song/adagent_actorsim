
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

import copy,tempfile,unittest
pass  # Path is provided by the portable boundary above
from common_flow import ROOT,OUT,read,write,asset
from core import AgentError
from evidence import terminal_facts,collect,decorate,diagnostics
from causal_scope import prepare_scope,PROFILE,PROTECTED
from governance import PROFILES,minimality
from contracts import guarded_change

class TerminalScopeTests(unittest.TestCase):
 def actual(self):return OUT/'tasks/llm_s551_b003_e04_01dd588c/global/cases/cutin551/on/B/run_report.json'
 def test_actual_terminal_frame_has_the_unexecuted_rejection_and_actor_conflict(self):
  facts=terminal_facts(self.actual());self.assertEqual(facts['index'],109);self.assertTrue(facts['observation_without_executed_action']);self.assertEqual(facts['reason'],'no_jointly_verified_action')
  cert=facts['certificates']['nominal_certificate'];self.assertTrue(cert['road']['passed']);self.assertTrue(cert['dynamics_bounds_passed']);self.assertFalse(cert['actors']['passed']);self.assertLess(cert['actors']['per_actor_minimum_separation_m'][-1],0.)
 def test_actual_evidence_includes_terminal_and_exact_diagnostic_accepts_controls_index(self):
  champion=read(read(OUT/'handoff.json')['campaign']+'/state.json')['champion']
  cfg=read(OUT/'source_config.json');cfg['evidence_roots']=[]
  evidence=decorate(collect(cfg,champion,[]));terminal=[k for k in evidence['items'] if k.startswith('terminal_')];self.assertTrue(terminal)
  item=next(k for k,v in evidence['items'].items() if k.startswith('report_') and v['case']=='cutin551')
  result,ids=diagnostics(evidence,[dict(kind='trace_window',item_id=item,start=109,end=109)],80000)
  frame=result['items'][ids[0]]['facts'][0];self.assertEqual(frame['index'],109);self.assertEqual(frame['road_gate']['reason'],'no_jointly_verified_action');self.assertIn(frame['terminal_evidence_item'],result['items']);self.assertFalse(frame['road_gate']['failure_axes']['actor_forecast_passed'])
 def test_verified_actor_exhaustion_grants_only_registered_planner_scope(self):
  terminal=terminal_facts(self.actual());e=dict(items=dict(source={'editable_methods':['JointReference.rollout']},terminal_report=dict(facts=terminal,asset=terminal['receipt'],case='cutin551',episode_role='working_champion')))
  parent=dict(profile='v15_joint',code_pins={'parent':'hash'});state=dict(active_task_id='task',champion=copy.deepcopy(parent),usage={'charged_tokens':203406});round={'parent':parent}
  with tempfile.TemporaryDirectory() as d:
   prepare_scope(state,round,e,Path(d));self.assertEqual(round['parent']['profile'],PROFILE);self.assertEqual(set(e['items']['source']['editable_methods']),PROFILES[PROFILE]['methods']);self.assertFalse(set(PROTECTED)&PROFILES[PROFILE]['methods']);self.assertEqual(state['usage']['charged_tokens'],203406)
 def test_unsubstantiated_failure_does_not_expand_scope(self):
  e={'items':{'source':{'editable_methods':['JointReference.rollout']}}};state=dict(active_task_id='task',champion={'code_pins':{}});round={'parent':{'profile':'v15_joint'}}
  with tempfile.TemporaryDirectory() as d:prepare_scope(state,round,e,Path(d))
  self.assertEqual(round['parent']['profile'],'v15_joint')
 def test_real_guides_can_be_changed_but_certificates_and_multiple_mechanisms_cannot(self):
  before=(ROOT/'integrations/light-tracking-curvature-v15-r2/joint_safety.py').read_text();after=before.replace('return_length=max(8.,target*2.5)','return_length=max(8.,target*3.)')
  changed=guarded_change(before,after,PROFILES[PROFILE]['methods']);self.assertEqual(changed,['JointReference.guides']);minimality(before,after,changed,dict(maximum_changed_methods=1,maximum_changed_lines=80))
  bad=before.replace('passed=minimum>=.05','passed=minimum>=-.05')
  with self.assertRaises(AgentError):guarded_change(before,bad,PROFILES[PROFILE]['methods'])
  two=after.replace('float(cruise*.5)','float(cruise*.4)');changed=guarded_change(before,two,PROFILES[PROFILE]['methods'])
  with self.assertRaises(AgentError):minimality(before,two,changed,dict(maximum_changed_methods=1,maximum_changed_lines=80))

 def test_runtime_host_upgrade_keeps_immutable_task_and_rejects_unapproved_change(self):
  from unittest.mock import patch
  import causal_scope
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code=root/'host.py';code.write_text('# revised');task=root/'task.json';write(task,{'host_pins':{str(code):'old'}});original=asset(task)
   with patch.object(causal_scope,'OUT',root):
    with self.assertRaises(ValueError):causal_scope.runtime_spec(task)
    from common_flow import sha
    current={str(code):sha(code)};write(root/'revision.json',{'new_host_pins':current});write(root/'runtime_host_binding.json',dict(tasks={str(task.resolve()):asset(task)},revision=asset(root/'revision.json'),host_pins=current))
    spec=causal_scope.runtime_spec(task);self.assertEqual(spec['host_pins'],current);self.assertEqual(asset(task),original)
    code.write_text('# unauthorized')
    with self.assertRaises(ValueError):causal_scope.runtime_spec(task)
 def test_stop_review_is_bounded_and_charged_not_an_infinite_retry(self):
  import sys
  sys.path.insert(0,str(ROOT/'integrations/autonomous-patch-agent-v8/tests'))
  from helpers import FakeAdapter,config,proposal
  from engine import Engine
  from provider import FixtureProvider
  import json
  class Adapter(FakeAdapter):
   def review_stop(self,*a):return {'type':'terminal_stop_review','message':'Review actual terminal evidence'}
  def stop(messages,n):return proposal(_hugsim_expand(json.loads(messages[-1]['content']))['evidence'],kind='stop')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);provider=FixtureProvider(stop);engine=Engine(root/'campaign',config(),provider,Adapter(root));engine.initialize('bounded_stop');state=engine.run()
   self.assertEqual(state['status'],'model_stopped');self.assertEqual(provider.calls,2);self.assertEqual(state['usage']['api_requests'],2);self.assertEqual(len(state['history']),1)
