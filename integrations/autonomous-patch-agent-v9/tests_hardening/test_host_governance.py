
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

import copy,json,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests')]
from helpers import config,metrics,FakeAdapter,reply
from core import AgentError,write,read,asset
import convergence,search_governance as search,governance
from final_verdict import finalize,validate
from engine import Engine
from provider import FixtureProvider

def verdict(m,b=None,retained=True,motion=True):return finalize({'accepted':True,'reasons':[]},m,b or metrics(.26),1e-9,retained,{'passed':motion},{'synthetic':'hash'})
def setup(recovery=False):
 c=config();c['budgets'].update(max_rounds=32,max_gpu_attempts=100,max_api_requests=40,wall_seconds=18000)
 b=metrics(.26)
 if recovery:b['cutin551'].update(status='terminated',controls=21,observations=22,termination_reason='no_jointly_verified_action')
 p=dict(c['convergence'],allow_control_failure_baseline=recovery);ledger=convergence.initial('training',b,p,True)
 champ={'version':'seed','code':'synthetic','code_pins':{'synthetic':'hash'},'metrics':b}
 s={'active_task_id':'training','task_convergence':{'training':ledger},'champion':champ,'task_archive':[],'rounds_started':0,'started_epoch':None,'usage':{'api_requests':0,'charged_tokens':0,'gpu_attempts':0}}
 return c,s
class QualifiedTests(unittest.TestCase):
 def test_high_score_with_rc_drop_is_not_qualified(self):
  m=metrics(.31);m['original551']['score']['rc']=.299
  self.assertFalse(validate(verdict(m)))
 def test_opposite_scene_gains_never_cancel(self):
  b=metrics(.26);m=copy.deepcopy(b);m['original551']['score']['hdscore']+=.05;m['cutin551']['score']['hdscore']-=.05
  self.assertFalse(verdict(m)['qualified'])
 def test_qualification_cannot_be_rebound_to_other_code_or_metrics(self):
  m=metrics(.3);v=verdict(m);self.assertTrue(validate(v,m,{'synthetic':'hash'}));self.assertFalse(validate(v,m,{'synthetic':'other'}));self.assertFalse(validate(v,metrics(.4),{'synthetic':'hash'}));v['gates']['motion']['passed']=False;self.assertFalse(validate(v))
 def test_missing_rc_is_rejected(self):
  m=metrics(.3);del m['cutin551']['score']['rc'];self.assertFalse(verdict(m)['qualified'])
 def test_all_subgates_required(self):
  for retained,motion in [(False,True),(True,False)]:self.assertFalse(verdict(metrics(.3),retained=retained,motion=motion)['qualified'])
 def test_registry_never_promotes_unqualified_high_score(self):
  b=metrics(.26);parent={'pool_id':'seed','profile':'v15_joint','version':'seed','reference_metrics':b};pool={'seed':dict(parent,eligible=True,metrics=b,evaluated_repairs=0)}
  bundle={'round_id':'high_bad','code':'unused','release':'unused','release_name':'unused','candidate_pins':{}};m=metrics(.9);v=verdict(m,retained=False)
  governance.add_result(pool,parent,bundle,'unused',m,v);self.assertFalse(pool['high_bad']['eligible']);self.assertEqual(governance.best_score(pool),.26)
 def test_observed_research_frontier_cannot_be_selected_as_qualified_parent(self):
  b=metrics(.26);parent={'pool_id':'seed','profile':'v15_joint','version':'seed','reference_metrics':b};pool={'seed':dict(parent,eligible=True,metrics=b,evaluated_repairs=0)};bundle={'round_id':'bad','code':'unused','release':'unused','release_name':'unused','candidate_pins':{'synthetic':'hash'}}
  governance.add_result(pool,parent,bundle,'unused',metrics(.99),verdict(metrics(.99),retained=False));self.assertIn('bad',governance.frontier(pool));self.assertNotIn('bad',governance.frontier(pool,qualified_only=True));self.assertEqual(governance.best_score(pool),.26)
 def test_registry_missing_final_verdict_fails_closed(self):
  b=metrics(.26);parent={'pool_id':'seed','profile':'v15_joint','version':'seed','reference_metrics':b};pool={'seed':dict(parent,eligible=True,metrics=b,evaluated_repairs=0)};bundle={'round_id':'bad','code':'unused','release':'unused','release_name':'unused','candidate_pins':{}}
  governance.add_result(pool,parent,bundle,'unused',metrics(.9));self.assertFalse(pool['bad']['eligible'])
class SearchTests(unittest.TestCase):
 def add(self,c,s,i,m,v):return search.observe(s,c,str(i),{'version':str(i),'code_pins':{'synthetic':'hash'},'metrics':m},m,v)
 def test_eight_warmup_failures_do_not_consume_patience(self):
  c,s=setup(True)
  for i in range(8):self.add(c,s,i,metrics(.99),verdict(metrics(.99),retained=False))
  stage=search.ensure(s,c);self.assertEqual(stage['effective_evaluations'],8);self.assertEqual(stage['search_stagnation'],0);self.assertIsNone(stage['best_qualified_candidate']);self.assertTrue(stage['search_stopped']);self.assertFalse(s['task_convergence']['training']['converged'])
 def test_first_qualified_starts_patience_without_debit(self):
  c,s=setup(True)
  for i in range(3):self.add(c,s,i,metrics(.9),verdict(metrics(.9),retained=False))
  m=metrics(.27);self.add(c,s,3,m,verdict(m));stage=search.ensure(s,c);self.assertEqual(stage['phase'],'convergence');self.assertEqual(stage['search_stagnation'],0);self.assertEqual(stage['best_qualified_score'],.27)
 def test_three_rejected_high_scores_exhaust_search_not_qualification(self):
  c,s=setup()
  for i in range(3):self.add(c,s,i,metrics(.99),verdict(metrics(.99),retained=False))
  stage=search.ensure(s,c);self.assertEqual(stage['best_qualified_score'],.26);self.assertEqual(stage['best_observed_score'],.99);self.assertTrue(stage['search_stopped']);self.assertFalse(s['task_convergence']['training']['converged']);self.assertEqual(stage['best_qualified_candidate']['version'],'seed')
 def test_three_small_qualified_gains_are_patience_three(self):
  c,s=setup()
  for i,n in enumerate([.265,.269,.272]):self.add(c,s,i,metrics(n),verdict(metrics(n)))
  self.assertEqual(search.ensure(s,c)['search_stagnation'],3);self.assertAlmostEqual(search.ensure(s,c)['best_qualified_score'],.272)
 def test_point01_breakthrough_resets_only_if_qualified(self):
  c,s=setup();self.add(c,s,0,metrics(.265),verdict(metrics(.265)));self.add(c,s,1,metrics(.276),verdict(metrics(.276)));self.assertEqual(search.ensure(s,c)['search_stagnation'],0)
 def test_retry_is_idempotent_and_conflicting_retry_rejected(self):
  c,s=setup();m=metrics(.265);v=verdict(m);self.add(c,s,0,m,v);self.add(c,s,0,m,v);self.assertEqual(search.ensure(s,c)['effective_evaluations'],1)
  with self.assertRaises(AgentError):self.add(c,s,0,metrics(.266),verdict(metrics(.266)))
 def test_stage_change_does_not_reset_usage(self):
  c,s=setup();s['usage']['charged_tokens']=100;s['usage']['gpu_attempts']=2;old=search.ensure(s,c);s['task_archive']=[{'task_id':'old','convergence':{'experiment_fingerprint':'synthetic-old'},'champion':{'metrics':metrics(.26)}}];new=search.ensure(s,c);self.assertIsNot(old,new);self.assertEqual(s['usage']['charged_tokens'],100);self.assertEqual(s['usage']['gpu_attempts'],2)
 def test_infrastructure_quota_is_independent_and_idempotent(self):
  c,s=setup();self.assertTrue(search.infrastructure_failure(s,c,'x'));self.assertTrue(search.infrastructure_failure(s,c,'x'));self.assertTrue(search.infrastructure_failure(s,c,'y'));self.assertFalse(search.infrastructure_failure(s,c,'z'));stage=search.ensure(s,c);self.assertEqual(stage['effective_evaluations'],0);self.assertEqual(stage['search_stagnation'],0)
 def test_dynamic_cost_scales_with_all_baselines(self):
  self.assertEqual(search.resource_plan(4,4,3,60)['episodes'],19)
  self.assertEqual(search.resource_plan(8,4,3,60)['episodes'],35)
 def test_insufficient_shared_resources_do_not_auto_extend(self):
  c,s=setup(True);c['budgets']['max_gpu_attempts']=2;before=copy.deepcopy(c['budgets']);self.assertFalse(search.admission(s,c)['passed']);self.assertEqual(c['budgets'],before)
 def test_new_pool_quota_never_resets_on_task_change(self):
  c,s=setup()
  for x in ['a','b','c']:self.assertTrue(search.admit_validation_pool(s,c,x,3)['passed'])
  self.assertTrue(search.admit_validation_pool(s,c,'a',3)['passed']);self.assertFalse(search.admit_validation_pool(s,c,'d',3)['passed'])
class EngineIntegrationTests(unittest.TestCase):
 def test_registry_anchor_uses_highest_qualified_baseline(self):
  c,s=setup();s['registry']={'best':dict(s['champion'],version='best',eligible=True,metrics=metrics(.28))};self.assertEqual(search.ensure(s,c)['best_qualified_score'],.28)
 def test_engine_rejects_rc_drop_and_retains_qualified_best(self):
  class Adapter(FakeAdapter):
   def results(self,b,a):
    m,strict=super().results(b,a);m['original551']['score']['rc']=.29;return m,strict
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;e=Engine(root/'campaign',c,FixtureProvider(reply),Adapter(root,[.35]));e.initialize('synthetic');s=e.run();self.assertEqual(s['history'][0]['status'],'rejected');self.assertEqual(s['champion']['version'],'fixture_seed');stage=s['search_stages'][s['active_search_stage']];self.assertEqual(stage['best_qualified_score'],.26)

class HistoricalRCTests(unittest.TestCase):
 def test_explicit_rc_drop_in_old_task_is_rejected(self):
  import task_rotation as rotation
  from governance import SHARED
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=metrics(.26)
   for name,row in m.items():
    file=root/(name+'.json');write(file,dict(**{k:k for k in SHARED},fresh_history=True,score_kind='executed_path'));row['contract']=asset(file)
    for kind in ['report','summary']:
     file=root/(name+kind+'.json');write(file,{});row[kind]=asset(file)
   ledger=convergence.initial('old',m,config()['convergence'],False);archive={'task_id':'old','convergence':ledger,'champion':{'metrics':m,'reference_metrics':m}}
   n=copy.deepcopy(m);n['cutin551']['score']['rc']-=.001;file=root/'proof.json';write(file,{'candidate_pins':{'code':'hash'},'task_id':'old'})
   class Adapter:
    def retention_check(self,*args):return {'passed':True,'candidate_pins':{'code':'hash'},'tasks':{'old':{'evaluation':asset(file),'metrics':n}}}
   self.assertFalse(rotation.verify_retention(Adapter(),{'candidate_pins':{'code':'hash'}},{},[archive],config()['acceptance'],False))
 def test_qualified_metric_mutation_invalidates_registry_anchor(self):
  m=metrics(.3);parent={'pool_id':'seed','profile':'v15_joint','version':'seed','reference_metrics':metrics(.26)};pool={'seed':dict(parent,eligible=True,metrics=metrics(.26),evaluated_repairs=0)};b={'round_id':'good','code':'unused','release':'unused','release_name':'unused','candidate_pins':{'synthetic':'hash'}}
  governance.add_result(pool,parent,b,'unused',m,verdict(m));self.assertEqual(governance.best_score(pool),.3);pool['good']['metrics']['cutin551']['score']['hdscore']=.99;self.assertEqual(governance.best_score(pool),.26)

class ResourceCalibrationTests(unittest.TestCase):
 def test_unknown_adapter_does_not_assume_free_batching(self):
  self.assertEqual(search.resource_plan(4,4,3,60)['gpu_dispatches'],19);self.assertEqual(search.resource_plan(4,4,3,60,training_episodes_per_dispatch=2)['gpu_dispatches'],11)
 def test_measured_baseline_cost_overrides_optimistic_fallback(self):
  c,s=setup();c['host_governance']['resource_model']['seconds_per_episode']=1.
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'summary';write(f,{'wall_seconds':800.});s['champion']['metrics']['original551']['summary']=asset(f);r=search.admission(s,c);self.assertGreaterEqual(r['minimum_resource_plan']['wall_seconds'],800*9)

class CounterexampleIcebreakingTests(unittest.TestCase):
 def test_score_regression_counterexample_starts_unqualified_even_if_run_completed(self):
  c,s=setup();s['counterexample_training_task_id']='training';stage=search.ensure(s,c);self.assertIsNone(stage['best_qualified_candidate']);self.assertEqual(stage['phase'],'icebreaking')
 def test_first_qualified_recovery_is_not_a_plateau_step(self):
  c,s=setup(True);m=metrics(.265);v=verdict(m);search.observe(s,c,'first',{'version':'first','code_pins':{'synthetic':'hash'},'metrics':m},m,v)
  row=convergence.observe(s['task_convergence']['training'],'first',m,v,.26,True);self.assertEqual(search.ensure(s,c)['search_stagnation'],0);self.assertEqual(s['task_convergence']['training']['small_gain_streak'],0);self.assertTrue(row['new_complete_baseline_established'])
