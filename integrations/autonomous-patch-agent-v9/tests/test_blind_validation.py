
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

import copy,json,subprocess,sys,tempfile,time,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
from helpers import config,metrics
from core import AgentError,asset,sha,read,write
import blind_validation as b
from privacy import public_evidence
from validation_privacy import validate_test
import generalization as g

class BlindValidationTests(unittest.TestCase):
 def setup(self,root,ready=False):
  private=root/'registry_private';private.mkdir();parent=root/'parent.py';parent.write_text('# parent');candidate=root/'candidate.py';candidate.write_text('# candidate');pins={str(parent):sha(parent)}
  def case(i):
   write(private/('baseline'+str(i)+'.json'),{'code_pins':pins})
   return dict(id='PRIVATE_CANARY_CASE_'+str(i),scene=i,ready=ready,baseline=asset(private/('baseline'+str(i)+'.json')) if ready else None,adapter={'module':asset(parent),'factory':'unused'} if ready else None)
  registry=dict(baseline_controller_pins=pins,development_cases=[case(901),case(902)],sealed_cases=[case(903)])
  write(private/'registry.json',registry);write(root/'global.json',{'version':'parent'})
  cfg=config();cfg['generalization']=dict(required=True,mode='blind_cross_scene',training_scene=551,private_registry=asset(private/'registry.json'),global_parent=asset(root/'global.json'),feedback_policy='binary_only',max_development_queries=3)
  state=dict(champion=dict(version='candidate',code_pins={str(candidate):sha(candidate)},metrics=metrics(.27)),active_task_id='training',current_task_descriptor={'scene':551},simulation_only=False,task_convergence={'training':dict(converged=True,window=[1,2,3],small_gain_streak=3,observations=[],policy={'target_case':'cutin551'})},usage=dict(api_requests=9,charged_tokens=203406,reported_tokens=203406,gpu_attempts=17,gpu_dispatches=17),started_epoch=time.time(),rounds_started=4,task_switch_pending=True)
  return cfg,state,registry
 def test_missing_private_assets_spends_nothing_and_exposes_no_case(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root);usage=copy.deepcopy(s['usage'])
   self.assertFalse(b.step(s,cfg,object(),root));self.assertEqual(s['status'],'waiting_private_validation_assets');self.assertEqual(s['usage'],usage);self.assertNotIn('PRIVATE_CANARY',json.dumps(s));self.assertFalse((root/'accepted_patch.json').exists())
 def test_private_rejection_returns_only_binary_feedback_and_preserves_budget(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root,True)
   def fail(shadow,*args):
    shadow['usage']['gpu_attempts']+=2;shadow['usage']['gpu_dispatches']+=2;shadow['generalization']={'status':'failed','results':[{'case_id':'PRIVATE_CANARY_CASE_901','reasons':['PRIVATE_CANARY_CAUSE'],'score':.123}]};return False
   with patch.object(g,'step',side_effect=fail):self.assertIsNone(b.step(s,cfg,object(),root))
   self.assertFalse(s['task_convergence']['training']['converged']);self.assertEqual(s['usage']['charged_tokens'],203406);self.assertEqual(s['usage']['gpu_attempts'],19);self.assertTrue(s['blind_repair']['audit_required']);self.assertEqual(b.public_feedback(s)['status'],'not_passed');self.assertNotIn('PRIVATE_CANARY',json.dumps(s));self.assertFalse((root/'accepted_patch.json').exists())
 def test_final_fold_cannot_be_reused_after_a_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root,True)
   def passed(shadow,*a):shadow['generalization']={'status':'passed'};return True
   def failed(shadow,*a):shadow['generalization']={'status':'failed'};return False
   with patch.object(g,'step',side_effect=passed):self.assertIsNone(b.step(s,cfg,object(),root))
   self.assertFalse((root/'accepted_patch.json').exists())
   with patch.object(g,'step',side_effect=failed):self.assertIsNone(b.step(s,cfg,object(),root))
   s['task_convergence']['training']['converged']=True
   with patch.object(g,'step') as gate:self.assertFalse(b.step(s,cfg,object(),root));gate.assert_not_called()
   self.assertEqual(s['status'],'waiting_fresh_private_validation_pool')
 def test_development_queries_are_bounded_without_resetting_API_budget(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root,True);cfg['generalization']['max_development_queries']=1
   def failed(shadow,*a):shadow['generalization']={'status':'failed'};return False
   with patch.object(g,'step',side_effect=failed):b.step(s,cfg,object(),root)
   s['task_convergence']['training']['converged']=True
   with patch.object(g,'step') as gate:self.assertFalse(b.step(s,cfg,object(),root));gate.assert_not_called()
   self.assertEqual(s['status'],'private_validation_query_budget_exhausted');self.assertEqual(s['usage']['api_requests'],9)
 def test_training_verdict_allows_equal_score_but_never_weakens_safety_or_scores(self):
  state=dict(blind_repair={'active':True},active_task_id='t',task_convergence={'t':{'policy':{'target_case':'cutin551'}}},champion={'metrics':metrics(.27)})
  v=dict(accepted=False,reasons=['cutin551:no_incremental_target_gain'])
  self.assertTrue(b.training_verdict(v,state,metrics(.27),{'numeric_tolerance':1e-9})['accepted'])
  unsafe=dict(v,reasons=v['reasons']+['cutin551:candidate_safety_or_stop_failure']);self.assertFalse(b.training_verdict(unsafe,state,metrics(.27),{'numeric_tolerance':1e-9})['accepted'])
  self.assertFalse(b.training_verdict(v,state,metrics(.269),{'numeric_tolerance':1e-9})['accepted'])
 def test_model_payload_filters_private_canaries_but_retains_training_data(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root)
   evidence=dict(items=dict(source={'text':'public controller','editable_methods':['JointReference.guides']},private_probe={'facts':'PRIVATE_CANARY_CAUSE'},report_0={'asset':{'path':str(root/'private_validation/results.json')},'facts':{'cause':'PRIVATE_CANARY_CAUSE'}},evaluation_0={'facts':{'missing_cases':['PRIVATE_CANARY_CASE_901'],'training_score':.27}}),history=[{'generalization':{'cause':'PRIVATE_CANARY_CAUSE'},'status':'accepted'}],task_contract={'case_ids':['original551','cutin551']})
   result=public_evidence(evidence,s,cfg,root);encoded=json.dumps(result);self.assertNotIn('PRIVATE_CANARY',encoded);self.assertIn('public controller',encoded);self.assertEqual(result['items']['evaluation_0']['facts']['training_score'],.27)
 def test_private_test_reads_and_symlink_alias_are_denied_in_a_separate_process(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);private=root/'private';private.mkdir();secret=private/'secret.txt';secret.write_text('PRIVATE_CANARY_CAUSE');alias=root/'alias.txt';alias.symlink_to(secret)
   agent=Path(__file__).resolve().parents[1]
   for path in [secret,alias]:
    script='import sys;sys.path.insert(0,'+repr(str(agent))+');from validation_privacy import install;install(['+repr(str(private))+']);open('+repr(str(path))+').read()'
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    self.assertNotEqual(result.returncode,0);self.assertIn('Private validation access blocked',result.stderr);self.assertNotIn('PRIVATE_CANARY_CAUSE',result.stdout+result.stderr)
 def test_exfiltration_imports_are_not_valid_new_causal_tests(self):
  with self.assertRaises(AgentError):validate_test("import ctypes\n")
  self.assertTrue(validate_test("import unittest\nclass C(unittest.TestCase):\n def test_x(self):self.assertTrue(True)\n"))
 def test_private_folds_must_be_disjoint_from_training_and_each_other(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root);r['sealed_cases'][0]['scene']=551;write(Path(cfg['generalization']['private_registry']['path']),r);cfg['generalization']['private_registry']=asset(Path(cfg['generalization']['private_registry']['path']))
   with self.assertRaises(AgentError):b.policy(cfg)

 def test_private_executor_stdout_and_exceptions_cannot_leak_to_coordinator(self):
  import io,contextlib
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,r=self.setup(root,True);output=io.StringIO()
   def fail(*a):print('PRIVATE_CANARY_CAUSE');raise RuntimeError('PRIVATE_CANARY_CAUSE')
   with patch.object(g,'step',side_effect=fail),contextlib.redirect_stdout(output):self.assertFalse(b.step(s,cfg,object(),root))
   self.assertNotIn('PRIVATE_CANARY',output.getvalue()+json.dumps(s));self.assertEqual(s['status'],'private_validation_infrastructure_error');self.assertEqual(s['usage']['api_requests'],9)
