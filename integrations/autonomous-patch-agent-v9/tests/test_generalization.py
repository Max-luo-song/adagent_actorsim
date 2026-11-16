
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

import copy,tempfile,unittest,time
pass  # Path is provided by the portable boundary above
from helpers import config,metric,write,asset
from core import AgentError,read
import generalization as g

class GeneralizationTests(unittest.TestCase):
 def fixture(self,root,scene=172,score=.3):
  code=root/'controller.py';code.write_text('# frozen controller\n');from core import sha
  pins={str(code):sha(code)};shared={k:asset(code) for k in ['scene_manifest','scenario','model_weights','static_map','scorer','score_config','physics']}
  shared.update(scene=scene,initial_context={'source_frame':10},source_end_frame=35,seed=0);write(root/'shared.json',shared);common=asset(root/'shared.json')
  def episode(tag,value):
   d=root/tag;d.mkdir();c={'scene':scene,'comparison_contract':common,'code_pins':pins,'fresh_history':True,'score_kind':'executed_path','history_run_id':tag,'controls':25};write(d/'contract.json',c)
   write(d/'plan.json',{'comparison_contract':common})
   # Synthetic dense motion evidence is now required by the v9 host gate.
   import numpy as np
   dense=np.c_[np.arange(31)*.1,np.zeros((31,2)),np.ones(31),np.zeros(31)].tolist()
   trace=[]
   for i in range(25):
    pose=np.eye(4);pose[0,3]=i*.1
    trace.append({'timestamp_s':i*.1,'ego_to_world':pose.tolist(),'control':{'reference_states':dense},'action':{'acc_mps2':0.,'steer_rate_radps':0.}})
   write(d/'report.json',{'status':'completed','real_closed_loop':True,'model_weights_unchanged':True,'validation_contract':asset(d/'contract.json'),'experiment_plan':asset(d/'plan.json'),'started_epoch':time.time()+1,'trace':trace,'synthetic_fixture_only':True,'observations':[{'environment':{}} for _ in range(26)]})
   write(d/'summary.json',{'native_scores':{'scores':{'executed_path':{k:{'value':value if k in ['hdscore','rc'] else 1.} for k in g.SCORE_KEYS}}},'anomalies':{'environment_collision':False},'stop_recovery_unresolved':False})
   return {k:asset(d/(k+'.json')) for k in ['report','summary','contract']}
  baseline=episode('baseline',.3);candidate=episode('candidate',score)
  write(root/'baseline.json',{'code_pins':pins,'comparison_contract':common,'episode':baseline})
  case={'id':'original'+str(scene),'scene':scene,'ready':True,'baseline':asset(root/'baseline.json')}
  proof={'case_id':case['id'],'candidate_pins':pins,'baseline':case['baseline'],'comparison_contract':common,'candidate':candidate};write(root/'proof.json',proof)
  return {'code_pins':pins},case,asset(root/'proof.json'),proof
 def test_unchanged_and_improved_scores_pass(self):
  for score in [.3,.31]:
   with tempfile.TemporaryDirectory() as d:
    c,s,p,_=self.fixture(Path(d),score=score);self.assertTrue(g.verify_case(p,c,s,0,1e-9)['passed'])
 def test_each_scene_must_pass_not_average(self):
  with tempfile.TemporaryDirectory() as d:
   c,s,p,_=self.fixture(Path(d),score=.299);r=g.verify_case(p,c,s,0,1e-9);self.assertFalse(r['passed']);self.assertIn('hdscore_regression',r['reasons'])
 def test_wrong_controller_pins_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   c,s,p,_=self.fixture(Path(d));c['code_pins']={}
   with self.assertRaises(AgentError):g.verify_case(p,c,s,0,1e-9)
 def test_missing_or_reused_history_rejected(self):
  for fresh,history in [(False,'candidate'),(True,'baseline')]:
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);c,s,p,v=self.fixture(root);path=root/'candidate/contract.json';r=read(path);r.update(fresh_history=fresh,history_run_id=history);write(path,r);v['candidate']['contract']=asset(path)
    report=read(root/'candidate/report.json');report['validation_contract']=asset(path);write(root/'candidate/report.json',report);v['candidate']['report']=asset(root/'candidate/report.json');write(root/'proof.json',v)
    with self.assertRaises(AgentError):g.verify_case(asset(root/'proof.json'),c,s,0,1e-9)
 def test_old_candidate_episode_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   c,s,p,_=self.fixture(Path(d))
   with self.assertRaises(AgentError):g.verify_case(p,c,s,time.time()+100,1e-9)
 def test_unknown_scene_or_changed_inputs_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   c,s,p,_=self.fixture(Path(d));s['scene']=552
   with self.assertRaises(AgentError):g.verify_case(p,c,s,0,1e-9)
 def test_incomplete_controls_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c,s,p,v=self.fixture(root);path=root/'candidate/report.json';r=read(path);r['trace']=r['trace'][:-1];write(path,r);v['candidate']['report']=asset(path);write(root/'proof.json',v)
   with self.assertRaises(AgentError):g.verify_case(asset(root/'proof.json'),c,s,0,1e-9)
 def test_tampered_summary_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c,s,p,_=self.fixture(root);(root/'candidate/summary.json').write_text('{}')
   with self.assertRaises(AgentError):g.verify_case(p,c,s,0,1e-9)
 def test_missing_admission_waits_without_confirmation_or_budget_use(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c,case,p,_=self.fixture(root);state={'champion':c,'active_task_id':'task','simulation_only':False,'usage':{'gpu_dispatches':0}};cfg=config();cfg['generalization_fixture_bypass']=False
   self.assertFalse(g.step(state,cfg,object(),root));self.assertEqual(state['status'],'waiting_generalization_assets');self.assertFalse((root/'accepted_patch.json').exists());self.assertEqual(state['usage']['gpu_dispatches'],0)
 def test_missing_cases_cannot_finalize(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c,case,p,_=self.fixture(root);state={'champion':c,'active_task_id':'task','simulation_only':False};cfg=config();req=g.request(state,cfg,root)
   with self.assertRaises(AgentError):g.finalize(state,req,root)
   self.assertFalse((root/'accepted_patch.json').exists())
 def test_disabled_or_single_scene_policy_rejected(self):
  for patch in [{'required':False},{'cases':[{'id':'a','scene':172}]}]:
   cfg=config();cfg['generalization'].update(patch)
   with self.assertRaises(AgentError):g.policy(cfg)
 def test_only_complete_bound_proofs_create_confirmed_package(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir();candidate,case1,p1,_=self.fixture(a);c2,case2,p2,proof2=self.fixture(b,scene=552)
   # Both episodes must run the same selected policy bytes/path, not copied source with different pins.
   for side in ['baseline','candidate']:
    contract=b/side/'contract.json';v=read(contract);v['code_pins']=candidate['code_pins'];write(contract,v)
    report=b/side/'report.json';r=read(report);r['validation_contract']=asset(contract);write(report,r)
    if side=='candidate':proof2['candidate'].update(contract=asset(contract),report=asset(report))
   baseline=read(b/'baseline.json');baseline['code_pins']=candidate['code_pins'];baseline['episode'].update(contract=asset(b/'baseline/contract.json'),report=asset(b/'baseline/report.json'));write(b/'baseline.json',baseline);case2['baseline']=asset(b/'baseline.json');proof2.update(candidate_pins=candidate['code_pins'],baseline=case2['baseline']);write(b/'proof.json',proof2);p2=asset(b/'proof.json')
   cfg=config();cfg['generalization']['cases']=[case1,case2];state={'champion':candidate,'active_task_id':'task','simulation_only':False};req=g.request(state,cfg,root)
   results=[g.verify_case(p,candidate,c,req['requested_epoch'],1e-9) for p,c in [(p1,case1),(p2,case2)]];state['generalization']['results']=results
   state['simulation_only']=True
   with self.assertRaises(AgentError):g.finalize(state,req,root)
   self.assertFalse((root/'accepted_patch.json').exists());state['simulation_only']=False;g.finalize(state,req,root);self.assertTrue(read(root/'accepted_patch.json')['confirmed']);self.assertEqual(state['final_patch_status'],'confirmed')
class CoordinatorGeneralizationTests(unittest.TestCase):
 def test_real_coordinator_enters_generalization_before_task_rotation(self):
  from engine import Engine
  from helpers import FakeAdapter,reply,config
  from provider import FixtureProvider
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg=config();cfg['generalization_fixture_bypass']=False;cfg['budgets']['max_rounds']=6
   provider=FixtureProvider(reply);engine=Engine(root/'campaign',cfg,provider,FakeAdapter(root,[.261,.262,.263,.264]));engine.initialize('gate');state=engine.run()
   self.assertEqual(state['status'],'waiting_generalization_assets');self.assertEqual(provider.calls,3)
   self.assertEqual(state['active_task_id'],'scene551_cutin_v1');self.assertTrue((root/'campaign/generalization_request.json').exists());self.assertFalse((root/'campaign/accepted_patch.json').exists());self.assertEqual(state['final_patch_status'],'pending_generalization')
 def test_generalization_jobs_share_budget_and_missing_second_case_cannot_confirm(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);candidate,case,proof,_=GeneralizationTests().fixture(root)
   cfg=config();cfg['generalization_fixture_bypass']=False;cfg['generalization']['cases']=[case,{'id':'original552','scene':552,'ready':False}]
   state={'champion':candidate,'active_task_id':'task','simulation_only':False,'usage':{'gpu_dispatches':0,'gpu_attempts':0,'charged_tokens':456,'api_requests':7},'started_epoch':time.time(),'rounds_started':2}
   write(root/'admission.json',{'passed':True,'case_id':case['id'],'candidate_pins':candidate['code_pins'],'baseline':case['baseline']})
   class Adapter:
    def generalization_stage(self,*args):return {'admission':asset(root/'admission.json'),'inputs':{},'command':['fixture-not-executed'],'outputs':[],'timeout':10}
    def generalization_result(self,*args):return proof
   with patch.object(g.jobs,'launch',return_value={'status':'succeeded'}):self.assertIsNone(g.step(state,cfg,Adapter(),root))
   self.assertEqual(state['usage']['gpu_attempts'],1);self.assertEqual(state['usage']['gpu_dispatches'],1);self.assertEqual(state['usage']['charged_tokens'],456);self.assertEqual(state['usage']['api_requests'],7)
   self.assertFalse(g.step(state,cfg,Adapter(),root));self.assertEqual(len(state['generalization']['results']),1);self.assertFalse((root/'accepted_patch.json').exists())
if __name__=='__main__':unittest.main()
