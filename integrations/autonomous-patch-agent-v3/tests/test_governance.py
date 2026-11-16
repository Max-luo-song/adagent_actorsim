
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

import copy,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from helpers import LOCAL,metrics,config,proposal
from core import AgentError,asset
import governance as g
from contracts import decision,guarded_change
from validation import acceptance
class GovernanceTests(unittest.TestCase):
 def pool(self):
  reference=metrics(.26090653);reference['cutin551'].update(final_abs_lateral_m=1.818,peak_abs_lateral_m=4.087)
  def entry(key,score,final,peak,profile,eligible=True):
   m=copy.deepcopy(reference);m['cutin551'].update(final_abs_lateral_m=final,peak_abs_lateral_m=peak);m['cutin551']['score']['hdscore']=score
   return dict(version=key,pool_id=key,profile=profile,eligible=eligible,metrics=m,evaluated_repairs=0,reference_metrics=reference)
  pool={'v13':entry('v13',.26090653,1.818,4.087,'reference_only',False),'v15':entry('v15',.26533468,4.507,4.990,'v15_joint'),'v17':entry('v17',.26285492,1.774,4.087,'v17_spatial')}
  return pool,pool['v17'],reference
 def evidence(self,pool,champ,ref):
  ctx=g.context(pool,champ,ref);return dict(items={k:{} for k in ['source',*('registry_'+i for i in pool)]},iteration_context=ctx,base_pool={k:dict(profile=v['profile']) for k,v in pool.items()})
 def plan(self,ctx,base='v15'):
  return dict(base_id=base,strategy='minimal_repair',why_this_base='Preserve highest measured gain',preserve='tracking curvature and joint selection',fix_only='lateral side effect',compared_versions=ctx['frontier_ids'],protected_score=ctx['protected_target_hdscore'],rollback_evidence_ids=[])
 def test_v15_is_not_dropped_because_v17_is_newer_or_lower_offset(self):
  pool,champ,ref=self.pool();self.assertEqual(g.frontier(pool),['v15','v17']);self.assertEqual(g.select_parent(pool,champ,ref)['pool_id'],'v15')
 def test_lateral_side_effect_is_not_called_physical_failure(self):
  pool,champ,ref=self.pool();self.assertTrue(g.healthy(pool['v15']['metrics']));self.assertEqual(g.context(pool,champ,ref)['required_base'],'v15')
 def test_switching_to_v17_before_minimal_repair_is_blocked(self):
  pool,champ,ref=self.pool();e=self.evidence(pool,champ,ref);p=proposal(e);p['iteration_plan']=self.plan(e['iteration_context'],'v17')
  with self.assertRaises(AgentError):decision(p,e)
  p['iteration_plan']['base_id']='v15';decision(p,e)
 def test_cannot_ignore_high_score_in_plan_or_omit_its_comparator(self):
  pool,champ,ref=self.pool();e=self.evidence(pool,champ,ref)
  for kind in ('score','comparators'):
   p=self.plan(e['iteration_context'])
   if kind=='score':p['protected_score']=champ['metrics']['cutin551']['score']['hdscore']
   else:p['compared_versions']=['v17']
   with self.assertRaises(AgentError):g.validate_plan(p,e)
 def test_beating_v13_and_v17_does_not_hide_loss_against_v15(self):
  pool,champ,ref=self.pool();n=copy.deepcopy(champ['metrics']);n['cutin551']['score']['hdscore']=.263
  base=acceptance(n,champ['metrics'],ref,config()['acceptance'],{'status':'failed'})
  self.assertTrue(base['accepted']);r=g.retain_gains(base,n,pool['v15'],pool);self.assertFalse(r['accepted']);self.assertIn('lost_high_score_branch_gain',r['reasons'])
 def test_preserving_v15_score_and_repairing_offset_is_eligible(self):
  pool,champ,ref=self.pool();n=copy.deepcopy(champ['metrics']);n['cutin551']['score']['hdscore']=pool['v15']['metrics']['cutin551']['score']['hdscore']
  base=acceptance(n,champ['metrics'],ref,config()['acceptance'],{'status':'failed'});self.assertTrue(g.retain_gains(base,n,pool['v15'],pool)['accepted'])
 def test_tradeoff_stays_in_frontier_and_enables_evidence_based_route_change(self):
  pool,champ,ref=self.pool();n=copy.deepcopy(champ['metrics']);n['cutin551'].update(final_abs_lateral_m=1.5);n['cutin551']['score']['hdscore']=.264
  bundle=dict(round_id='trial',code='c',release='r',release_name='r',candidate_pins={})
  g.add_result(pool,pool['v15'],bundle,'m',n);self.assertIn('v15',g.frontier(pool));self.assertIn('trial',g.frontier(pool));self.assertEqual(pool['v15']['evaluated_repairs'],1)
  e=self.evidence(pool,champ,ref);self.assertIsNone(e['iteration_context']['required_base']);self.assertEqual(g.select_parent(pool,champ,ref)['pool_id'],'trial');p=self.plan(e['iteration_context'],'trial');p['strategy']='alternative_branch';p['rollback_evidence_ids']=['registry_v15']
  with self.assertRaises(AgentError):g.validate_plan(p,e)
  p['rollback_evidence_ids']=['registry_trial'];g.validate_plan(p,e)
 def test_original_regressing_branch_cannot_raise_best_score_target(self):
  pool,champ,ref=self.pool();n=copy.deepcopy(champ['metrics']);n['cutin551']['score']['hdscore']=.4;n['original551']['score']['hdscore']=.2
  g.add_result(pool,pool['v15'],dict(round_id='bad',code='c',release='r',release_name='r',candidate_pins={}), 'm',n)
  self.assertFalse(pool['bad']['eligible']);self.assertAlmostEqual(g.best_score(pool),.26533468)
 def test_minimal_patch_does_not_change_two_mechanisms(self):
  with self.assertRaises(AgentError):g.minimality('a','b',['one','two'],dict(maximum_changed_methods=1,maximum_changed_lines=80))
 def test_v15_profile_preserves_process_and_original_curvature_generation(self):
  old=(LOCAL.parent/'light-tracking-curvature-v15-r2/joint_safety.py').read_text();before='return candidate if objective(candidate)<objective(baseline) else baseline';after='return candidate if objective(candidate)+.001<objective(baseline) else baseline'
  self.assertEqual(guarded_change(old,old.replace(before,after),g.PROFILES['v15_joint']['methods']),['JointReference.rollout'])
  with self.assertRaises(AgentError):guarded_change(old,old.replace('max_curv=max(abs(np.tan(s[4]))','max_curv=max(0.*abs(np.tan(s[4]))'),g.PROFILES['v15_joint']['methods'])
if __name__=='__main__':unittest.main()

class GovernedEngineTests(unittest.TestCase):
 def run_campaign(self,root,target):
  import json,shutil
  from helpers import FakeAdapter,reply
  from engine import Engine
  from provider import FixtureProvider
  class Adapter(FakeAdapter):
   def registered_pool(self,c,champ):
    high=copy.deepcopy(champ);high.update(version='v15',pool_id='v15',profile='v15_joint',eligible=True,evaluated_repairs=0)
    old=Path(champ['code']);new=old.parent/'seed15';shutil.copytree(old,new);high.update(code=str(new))
    from core import code_pins
    high['code_pins']=code_pins(new);high['metrics']['cutin551'].update(final_abs_lateral_m=4.5,peak_abs_lateral_m=5.);high['metrics']['cutin551']['score']['hdscore']=.265
    stable=copy.deepcopy(champ);stable.update(pool_id='v17',profile='v17_spatial',eligible=True,evaluated_repairs=0)
    return {'v15':high,'v17':stable}
  def fixture(messages,n):
   e=_hugsim_expand(json.loads(messages[-1]['content']))['evidence'];p=proposal(e);ctx=e['iteration_context']
   p['iteration_plan']=dict(base_id='v17' if n==1 else e['selected_parent_id'],strategy='minimal_repair',why_this_base='Retain high-score gain',preserve='v15 curvature and joint selection',fix_only='lateral side effect',compared_versions=ctx['frontier_ids'],protected_score=ctx['protected_target_hdscore'],rollback_evidence_ids=[])
   return p
  c=config();c['budgets']['max_rounds']=1;provider=FixtureProvider(fixture);adapter=Adapter(root,[target]);engine=Engine(root/'campaign',c,provider,adapter);engine.initialize('governed');state=engine.run();return state,provider
 def test_engine_blocks_lower_base_and_retains_score_quality_tradeoff(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s,p=self.run_campaign(root,.263)
   self.assertEqual(p.calls,2);self.assertEqual(s['history'][0]['status'],'tradeoff_retained');self.assertEqual(s['champion']['version'],'fixture_seed');self.assertIn('v15',s['registry']);self.assertIn('governed_r001',s['registry']);self.assertEqual(s['registry']['v15']['evaluated_repairs'],1)
   selection=__import__('json').loads((root/'campaign/rounds/governed_r001/base_selection_2.json').read_text());self.assertEqual(selection['plan']['base_id'],'v15')
 def test_engine_promotes_only_when_best_score_and_quality_both_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);s,p=self.run_campaign(root,.266)
   self.assertEqual(s['history'][0]['status'],'accepted');self.assertEqual(s['champion']['profile'],'v15_joint');self.assertEqual(s['champion']['pool_id'],'governed_r001');self.assertFalse(s['production_driver_changed'])
