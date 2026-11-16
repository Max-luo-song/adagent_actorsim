
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
from helpers import config,metrics,FakeAdapter,reply
from core import AgentError,write,read,asset
from convergence import initial,observe
from engine import Engine
from provider import FixtureProvider
import task_rotation as rotation
class PlateauTests(unittest.TestCase):
 def setUp(self):self.policy=config()['convergence'];self.base=metrics(.26)
 def add(self,ledger,rid,before,score,accepted=True,reasons=None):return observe(ledger,rid,metrics(score),dict(accepted=accepted,reasons=reasons or []),before,True)
 def test_exact_user_rule_three_valid_gains_below_point01(self):
  ledger=initial('task',self.base,self.policy,True);before=.26
  for i,gain in enumerate((.005,.004,.003)):
   self.add(ledger,str(i),before,before+gain);before+=gain
  self.assertTrue(ledger['converged']);self.assertEqual(ledger['small_gain_streak'],3);self.assertAlmostEqual(ledger['best_score'],.272)
 def test_material_gain_and_threshold_equality_reset_streak(self):
  ledger=initial('task',self.base,self.policy,True);self.add(ledger,'a',.26,.265);self.add(ledger,'b',.265,.275);self.assertEqual(ledger['small_gain_streak'],0);self.add(ledger,'c',.275,.279);self.assertFalse(ledger['converged'])
 def test_nonfinite_baseline_is_not_admitted(self):
  m=metrics(.26);m['original551']['progress_m']=float('nan')
  with self.assertRaises(AgentError):initial('task',m,self.policy,True)
 def test_zero_gain_valid_nonregression_can_converge(self):
  ledger=initial('task',self.base,self.policy,True)
  for i in range(3):self.add(ledger,str(i),.26,.26,False,['cutin551:no_incremental_target_gain'])
  self.assertTrue(ledger['converged'])
 def test_safety_or_original_regression_is_not_convergence(self):
  ledger=initial('task',self.base,self.policy,True);self.add(ledger,'a',.26,.261);self.add(ledger,'b',.261,.262);self.add(ledger,'bad',.262,.263,False,['original551:original_hd_regression']);self.assertFalse(ledger['converged']);self.assertEqual(ledger['small_gain_streak'],0)
 def test_duplicate_is_idempotent_conflicting_duplicate_rejected(self):
  ledger=initial('task',self.base,self.policy,True);self.add(ledger,'a',.26,.261);self.add(ledger,'a',.26,.261);self.assertEqual(ledger['small_gain_streak'],1)
  with self.assertRaises(AgentError):self.add(ledger,'a',.26,.262)
 def test_new_task_uses_own_baseline_and_restarts_window(self):
  ledger=initial('other_task',metrics(.7),self.policy,True);self.add(ledger,'a',.7,.705);self.assertAlmostEqual(ledger['window'][0]['effective_gain'],.005);self.assertEqual(ledger['small_gain_streak'],1)
 def test_different_seed_or_edit_contract_cannot_share_window(self):
  from governance import SHARED
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);base=metrics(.26)
   for name,m in base.items():
    path=folder/(name+'.json');write(path,dict(**{k:k for k in SHARED},fresh_history=True,score_kind='executed_path'));m['contract']=asset(path)
   ledger=initial('task',base,self.policy,False);candidate=copy.deepcopy(base);changed=folder/'other.json';contract=read(base['cutin551']['contract']['path']);contract['seed']='different';write(changed,contract);candidate['cutin551']['contract']=asset(changed)
   with self.assertRaisesRegex(AgentError,'Different scenario/edit/seed'):observe(ledger,'a',candidate,dict(accepted=True,reasons=[]),.26,False)

class TaskAdapter(FakeAdapter):
 def activate_task(self,task,carry,archive,budget):
  self.activation_count=getattr(self,'activation_count',0)+1;admission=self.root/'next_task_admission.json';write(admission,{'task_id':task['id']});champ=copy.deepcopy(carry);champ.update(version='next_task_baseline',metrics=metrics(.5),reference_metrics=metrics(.5))
  return dict(task_id=task['id'],admission=asset(admission),champion=champ,reference_metrics=metrics(.5),registry=None,target_case='cutin551',validated=True)
 def verify_task_admission(self,task,activation):return True
 def retention_check(self,bundle,attempt,archives):return True

class RotationEngineTests(unittest.TestCase):
 def test_production_retention_rejects_nonfinite_progress(self):
  from governance import SHARED
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=metrics(.26)
   for name,row in m.items():
    contract=root/(name+'.json');write(contract,dict(**{k:k for k in SHARED},fresh_history=True,score_kind='executed_path'));row['contract']=asset(contract)
    for kind in ('report','summary'):
     path=root/(name+kind+'.json');write(path,{});row[kind]=asset(path)
   ledger=initial('old',m,config()['convergence'],False);archive={'task_id':'old','convergence':ledger,'champion':{'metrics':m,'reference_metrics':m}};n=copy.deepcopy(m);n['cutin551']['progress_m']=float('nan');evaluation=root/'receipt.json';write(evaluation,{'candidate_pins':{'code':'hash'},'task_id':'old'})
   class Adapter:
    def retention_check(self,*args):return {'passed':True,'candidate_pins':{'code':'hash'},'tasks':{'old':{'evaluation':asset(evaluation),'metrics':n}}}
   self.assertFalse(rotation.verify_retention(Adapter(),{'candidate_pins':{'code':'hash'}},{},[archive],config()['acceptance'],False))
 def test_retention_false_dictionary_and_unverified_production_bool_are_rejected(self):
  class FalseResult:
   def retention_check(self,*args):return {'passed':False}
  class Unverified:
   def retention_check(self,*args):return True
  self.assertFalse(rotation.verify_retention(FalseResult(),{}, {}, [],config()['acceptance'],True));self.assertFalse(rotation.verify_retention(Unverified(),{}, {}, [],config()['acceptance'],False))
 def test_host_task_case_names_replace_initial_labels_without_allowing_model_edits(self):
  from helpers import proposal
  from contracts import decision
  e={'items':{'source':{}},'task_contract':{'case_ids':['original552','edited552']}};p=proposal(e);p['edit_instruction']['cases']=['original552','edited552'];decision(p,e);p['edit_instruction']['cases']=['original551','cutin551']
  with self.assertRaises(AgentError):decision(p,e)
 def test_registered_adapter_hash_binding_and_resume_factory(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);module=root/'adapter.py';module.write_text('from test_convergence_rotation import TaskAdapter\nfrom validation import acceptance as judge\nfrom evidence import collect\nclass Loaded(TaskAdapter):\n acceptance=staticmethod(judge)\n collect_evidence=staticmethod(collect)\ndef make_adapter(task,config,checkpoint):return Loaded(task["test_root"])\n')
   spec={'id':'bound','scene':551,'edit':'new','ready':True,'test_root':str(root),'adapter':{'module':asset(module),'factory':'make_adapter'}};c=config();c['task_queue']['tasks'].append(spec);e=Engine(root/'campaign',c,FixtureProvider(reply),FakeAdapter(root));state=e.initialize('binding');state['active_task_spec']=spec;e.save(state);replacement=Engine(root/'campaign',provider=FixtureProvider(reply));self.assertEqual(type(replacement.adapter).__name__,'Loaded');module.write_text(module.read_text()+'# changed\n')
   with self.assertRaises(AgentError):Engine(root/'campaign',provider=FixtureProvider(reply))

 def setup_engine(self,folder,adapter_type=TaskAdapter,rounds=4):
  c=config();c['budgets']['max_rounds']=rounds;c['task_queue']={'current_id':'first','tasks':[dict(id='first',scene=551,edit='a',ready=True),dict(id='second',scene=551,edit='b',ready=True)]};a=adapter_type(folder,[.261,.262,.263,.501]);p=FixtureProvider(reply);e=Engine(folder/'campaign',c,p,a);e.initialize('rotation');return e,a,p,c
 def test_three_small_patches_archive_and_automatically_start_admitted_next_task(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e,a,p,c=self.setup_engine(root);s=e.run();self.assertEqual(s['active_task_id'],'second');self.assertEqual(a.activation_count,1);self.assertEqual(len(s['history']),4);self.assertEqual(s['usage']['api_requests'],4);self.assertEqual(s['rounds_started'],4);self.assertEqual(s['task_archive'][0]['task_id'],'first');self.assertAlmostEqual(s['task_archive'][0]['champion']['metrics']['cutin551']['score']['hdscore'],.263);self.assertAlmostEqual(s['task_convergence']['second']['window'][0]['effective_gain'],.001);self.assertTrue((root/'campaign/task_transitions/first_to_second.json').exists())
 def test_no_admitted_task_waits_and_keeps_accepted_patch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e,a,p,c=self.setup_engine(root,FakeAdapter,rounds=6);s=e.run();self.assertEqual(s['status'],'waiting_next_task');self.assertEqual(p.calls,3);self.assertAlmostEqual(s['champion']['metrics']['cutin551']['score']['hdscore'],.263);self.assertTrue((root/'campaign/next_task_request.json').exists())
 def test_previous_task_check_cannot_be_silently_skipped(self):
  class NoRetention(TaskAdapter):
   def retention_check(self,*args):return False
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e,a,p,c=self.setup_engine(root,NoRetention);s=e.run();self.assertEqual(s['history'][-1]['status'],'rejected');self.assertIn('previous_task_nonregression_unverified',s['history'][-1]['acceptance']['reasons']);self.assertAlmostEqual(s['champion']['metrics']['cutin551']['score']['hdscore'],.5)
 def test_task_switch_does_not_reset_shared_budget(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);e,a,p,c=self.setup_engine(root,rounds=3);s=e.run();self.assertEqual(s['status'],'budget_exhausted');self.assertFalse(hasattr(a,'activation_count'));self.assertEqual(s['usage']['api_requests'],3);self.assertEqual(s['active_task_id'],'first')
 def test_api_and_cpu_failures_do_not_count_as_plateau(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),FakeAdapter(root,fail_cpu=True));e.initialize('failure');s=e.run();ledger=s['task_convergence'][s['active_task_id']];self.assertEqual(ledger['small_gain_streak'],0);self.assertEqual(ledger['observations'],[])
if __name__=='__main__':unittest.main()
