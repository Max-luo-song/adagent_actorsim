
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
from helpers import config,metrics
from core import AgentError,asset,read,write,sha,digest
import generalization as g
import validation_lifecycle as lifecycle
from privacy import public_evidence
import test_generalization as fixtures

class ValidationLifecycleTests(unittest.TestCase):
 def setup(self,root):
  private=root/'host_private';private.mkdir()
  cases=[];proofs=[];candidate=None
  for scene in (901,902):
   folder=private/str(scene);folder.mkdir();c,case,binding,proof=fixtures.GeneralizationTests().fixture(folder,scene,score=.29 if scene==901 else .31)
   if candidate is None:candidate=c
   else:
    for name in ('baseline','candidate'):
     file=folder/name/'contract.json';contract=read(file);contract['code_pins']=candidate['code_pins'];write(file,contract)
     file=folder/name/'report.json';report=read(file);report['validation_contract']=asset(folder/name/'contract.json');write(file,report)
    base=read(folder/'baseline.json');base['code_pins']=candidate['code_pins'];base['episode'].update(contract=asset(folder/'baseline/contract.json'),report=asset(folder/'baseline/report.json'));write(folder/'baseline.json',base)
    case['baseline']=asset(folder/'baseline.json');proof.update(candidate_pins=candidate['code_pins'],baseline=case['baseline']);proof['candidate'].update(contract=asset(folder/'candidate/contract.json'),report=asset(folder/'candidate/report.json'));write(folder/'proof.json',proof);binding=asset(folder/'proof.json')
   case['ready']=True;case['adapter']={'module':asset(private/'901/controller.py'),'factory':'unused'};cases.append(case);proofs.append(binding)
  pool={'cases':cases,'baseline_controller_pins':candidate['code_pins']};write(private/'pool.json',pool);binding=asset(private/'pool.json');write(private/'catalog.json',{'pools':[binding]})
  cfg={'generalization':{'mode':lifecycle.MODE,'feedback_policy':'retire_then_release','required':True,'pool_catalog':asset(private/'catalog.json'),'training_scene':551}}
  state={'champion':candidate,'active_task_id':'training','current_task_descriptor':{'scene':551},'usage':{'charged_tokens':203406,'api_requests':9,'gpu_attempts':17},'task_convergence':{'training':{'converged':True}}}
  req={'cases':cases,'requested_epoch':0,'score_tolerance':1e-9}
  results=[g.verify_case(b,candidate,c,0,1e-9) for b,c in zip(proofs,cases)]
  return cfg,state,binding,pool,req,results
 def test_entire_pool_is_retired_before_cause_is_read_and_release_is_idempotent(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);usage=copy.deepcopy(s['usage']);original=lifecycle._cause
   def cause(side):
    _,ledger=lifecycle.load_ledger(cfg);self.assertEqual(ledger['pools'][digest(b)]['status'],'retired_failed');return original(side)
   with patch.object(lifecycle,'_cause',side_effect=cause):release=lifecycle.retire_and_release(s,cfg,root,b,p,q,r)
   _,ledger=lifecycle.load_ledger(cfg);self.assertEqual(ledger['consumed_scenes'],[901,902]);self.assertEqual(s['usage'],usage);self.assertEqual(len(s['pending_counterexamples']),1)
   self.assertEqual(lifecycle.retire_and_release(s,cfg,root,b,p,q,r),release);self.assertEqual(len(s['pending_counterexamples']),1);self.assertFalse((root/'accepted_patch.json').exists())
 def test_cause_read_crash_leaves_dataset_consumed_and_restart_can_release(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root)
   with patch.object(lifecycle,'_cause',side_effect=OSError('simulated crash')):
    with self.assertRaises(OSError):lifecycle.retire_and_release(s,cfg,root,b,p,q,r)
   self.assertEqual(lifecycle.choose_pool(s,cfg),(None,None))
   release=lifecycle.retire_and_release(s,cfg,root,b,p,q,r);self.assertEqual(lifecycle.verify_release(release,cfg)['pool_status'],'retired_failed')
 def test_only_retired_failures_are_disclosed_and_new_private_pool_stays_hidden(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);e={'items':{'source':{'text':'training code'},'report_hidden':{'asset':r[0]['proof'],'facts':{'reason':'ACTIVE_PRIVATE_CANARY'}}},'task_contract':{'case_ids':['training']}}
   self.assertNotIn('ACTIVE_PRIVATE_CANARY',json.dumps(public_evidence(e,s,cfg,root)))
   release=lifecycle.retire_and_release(s,cfg,root,b,p,q,r);out=public_evidence(e,s,cfg,root)
   self.assertIn('hdscore_regression',json.dumps(out));self.assertNotIn('ACTIVE_PRIVATE_CANARY',json.dumps(out));self.assertEqual(len(out['items']['released_counterexample_0']['facts']['failed_cases']),1)
   self.assertTrue(lifecycle.released_paths(cfg,[release]))
 def test_future_validation_excludes_all_old_scenes_even_renamed_cases(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);lifecycle.retire_and_release(s,cfg,root,b,p,q,r)
   old=copy.deepcopy(p);old['cases'][0]['id']='RENAMED';write(root/'host_private/renamed.json',old)
   fresh=copy.deepcopy(p)
   for i,c in enumerate(fresh['cases']):c.update(id='fresh_'+str(i),scene=903+i)
   write(root/'host_private/fresh.json',fresh);write(root/'host_private/catalog.json',{'pools':[asset(root/'host_private/renamed.json'),asset(root/'host_private/fresh.json')]});cfg['generalization']['pool_catalog']=asset(root/'host_private/catalog.json')
   chosen,pool=lifecycle.choose_pool(s,cfg);self.assertEqual({c['scene'] for c in pool['cases']},{903,904})
 def test_unretired_release_is_never_authorized(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);write(root/'fake.json',{'pool_identity':digest(b),'pool':b,'pool_status':'retired_failed','validation_reuse_forbidden':True})
   with self.assertRaises(AgentError):lifecycle.verify_release(asset(root/'fake.json'),cfg)
 def test_infrastructure_gate_does_not_release_or_reset_budget(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);other=root/'parent.py';other.write_text('#other');p['baseline_controller_pins']={str(other):sha(other)}
   for case in p['cases']:
    base=read(case['baseline']['path']);base['code_pins']=p['baseline_controller_pins'];write(case['baseline']['path'],base);case['baseline']=asset(case['baseline']['path'])
   write(b['path'],p);b=asset(b['path']);write(root/'host_private/catalog.json',{'pools':[b]});cfg['generalization']['pool_catalog']=asset(root/'host_private/catalog.json');usage=copy.deepcopy(s['usage']);s.update(started_epoch=0,rounds_started=4,simulation_only=False)
   def failed(shadow,*args):shadow['generalization']={'status':'invalid'};return False
   with patch.object(g,'step',side_effect=failed):self.assertFalse(lifecycle.step(s,cfg,object(),root))
   self.assertEqual(s['status'],'validation_infrastructure_error');self.assertEqual(s['usage'],usage);self.assertFalse(s.get('released_counterexample_manifests'))
 def test_verified_control_termination_is_failure_without_fake_full_horizon_score(self):
  for error in (None,'dependency missing'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);c,case,b,proof=fixtures.GeneralizationTests().fixture(root)
    file=root/'candidate/report.json';report=read(file);report.update(status='terminated',termination_reason='no_jointly_verified_action',trace=[{}]*9,observations=[{'environment':{}}]*10,error=error);write(file,report);proof['candidate']['report']=asset(file);write(root/'proof.json',proof)
    if error:
     with self.assertRaises(AgentError):g.verify_case(asset(root/'proof.json'),c,case,0,1e-9)
    else:
     result=g.verify_case(asset(root/'proof.json'),c,case,0,1e-9);self.assertFalse(result['passed']);self.assertIsNone(result['score_deltas']);self.assertEqual(result['candidate']['controls'],9)
 def test_no_scene_adapter_waits_instead_of_borrowing_training_scene(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);lifecycle.retire_and_release(s,cfg,root,b,p,q,r)
   class NoCompiler:pass
   with patch.object(lifecycle.PrivateAdapter,'_adapter',return_value=NoCompiler()):self.assertFalse(lifecycle.promote_pending(s,cfg,root))
   self.assertEqual(s['status'],'waiting_counterexample_training_admission');self.assertEqual(s['active_task_id'],'training')

 def test_counterexample_preparation_reserves_shared_GPU_budget_once(self):
  from core import BudgetExceeded
  import time
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);release=lifecycle.retire_and_release(s,cfg,root,b,p,q,r);cfg['budgets']={'max_gpu_attempts':17,'wall_seconds':3600};s['started_epoch']=time.time()
   admission=root/'prepare_admission.json';write(admission,{'passed':True,'case_id':p['cases'][0]['id'],'candidate_pins':s['champion']['code_pins'],'release':release})
   class Adapter:
    def counterexample_training_stage(self,*args):return {'admission':asset(admission),'inputs':{},'command':['registered_scene_worker'],'timeout':100,'outputs':[]}
    def counterexample_training_result(self,*args):raise AssertionError('not finished')
   with patch.object(lifecycle.PrivateAdapter,'_adapter',return_value=Adapter()),patch.object(lifecycle.jobs,'launch',return_value={'status':'running'}) as launch:
    with self.assertRaises(BudgetExceeded):lifecycle.promote_pending(s,cfg,root)
    launch.assert_not_called();cfg['budgets']['max_gpu_attempts']=18
    self.assertIsNone(lifecycle.promote_pending(s,cfg,root));self.assertIsNone(lifecycle.promote_pending(s,cfg,root));self.assertEqual(s['usage']['gpu_attempts'],18)
   self.assertEqual(s['usage']['charged_tokens'],203406)
 def test_training_activation_carries_policy_keeps_prior_task_and_does_not_reset_budget(self):
  import task_rotation as rotation
  from governance import SHARED
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg,s,b,p,q,r=self.setup(root);release=lifecycle.retire_and_release(s,cfg,root,b,p,q,r);usage=copy.deepcopy(s['usage'])
   code=root/'code';code.mkdir();file=code/'joint_safety.py';file.write_text('# carried policy');pins={str(file):sha(file)};champ={'code':str(code),'code_pins':pins,'version':'carried','metrics':metrics(.31),'reference_metrics':metrics(.31)}
   for name,m in champ['metrics'].items():
    folder=root/name;folder.mkdir();contract={**{k:k for k in SHARED},'scene':901,'code_pins':pins,'fresh_history':True,'score_kind':'executed_path'};write(folder/'contract.json',contract);m['contract']=asset(folder/'contract.json')
    write(folder/'report.json',{'validation_contract':m['contract'],'real_closed_loop':True,'model_weights_unchanged':True,'trace':[{}]*m['controls'],'observations':[{}]*m['observations']});write(folder/'summary.json',{})
    m.update(report=asset(folder/'report.json'),summary=asset(folder/'summary.json'))
   s.update(champion=champ,simulation_only=False,task_archive=[{'task_id':'training','confirmed':False}]);cfg['convergence']=config()['convergence']
   task={'id':'released_scene901','scene':901,'edit':'released_failure','ready':True,'adapter':{'module':asset(file),'factory':'unused'}};write(root/'admission.json',{'release':release,'scene':901,'edit':task['edit'],'fresh_baseline':True,'carried_policy_sha256':sha(file)})
   activation={'task_id':task['id'],'validated':True,'admission':asset(root/'admission.json'),'champion':champ,'reference_metrics':champ['reference_metrics'],'registry':None,'target_case':'cutin551'}
   class Adapter:
    def verify_task_admission(self,*args):return True
   with patch.object(rotation,'load_adapter',return_value=Adapter()):rotation.activate_counterexample(s,cfg,{'task':task,'activation':activation},release,p['cases'][0],root)
   self.assertEqual(s['active_task_id'],task['id']);self.assertEqual(s['champion']['code_pins'],pins);self.assertEqual(s['usage'],usage);self.assertEqual(s['task_archive'][0]['task_id'],'training');self.assertEqual(s['task_convergence'][task['id']]['small_gain_streak'],0);self.assertEqual(s['final_patch_status'],'provisional');self.assertTrue(rotation.verify_counterexample_task(s,cfg))
   s['active_task_spec']['scene']=902
   with self.assertRaises(AgentError):rotation.verify_counterexample_task(s,cfg)
