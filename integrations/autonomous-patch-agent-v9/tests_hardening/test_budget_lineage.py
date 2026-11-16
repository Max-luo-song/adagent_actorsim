
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
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from core import AgentError,write,asset
from budget_lineage import inherit
class LineageTests(unittest.TestCase):
 def test_missing_lineage_does_not_create_free_campaign(self):
  with self.assertRaises(AgentError):inherit({}, {'budgets':{}})
 def test_upgrade_carries_all_usage_and_original_clock(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);old={'usage':{'charged_tokens':42,'api_requests':3,'gpu_attempts':2},'rounds_started':4,'started_epoch':100.,'validation_pool_attempts':{'opaque':{'reserved':True}}};cfg={'budgets':{'max_rounds':12}}
   write(p/'state',old);write(p/'config',cfg);write(p/'proof',{'schema':'host_budget_lineage_v1','previous_state':asset(p/'state'),'previous_config':asset(p/'config')});new={};inherit(new,dict(cfg,budget_lineage=asset(p/'proof')))
   self.assertEqual(new['usage'],old['usage']);self.assertEqual(new['started_epoch'],100.);self.assertEqual(new['rounds_started'],4);self.assertEqual(len(new['validation_pool_attempts']),1)
   with self.assertRaises(AgentError):inherit({}, {'budgets':{'max_rounds':24},'budget_lineage':asset(p/'proof')})

class RealInitializationTests(unittest.TestCase):
 def test_real_campaign_cannot_use_legacy_public_fixture_pool(self):
  from unittest.mock import Mock
  from engine import Engine
  from provider import FixtureProvider
  import sys
  sys.path.insert(0,str(LOCAL/'tests'))
  from helpers import config,reply
  with tempfile.TemporaryDirectory() as d:
   adapter=Mock();adapter.simulation_only=False;cfg=config();cfg['generalization']['mode']='cross_scene';e=Engine(Path(d)/'new',cfg,FixtureProvider(reply),adapter)
   with self.assertRaises(AgentError):e.initialize('synthetic_guard')
   adapter.bootstrap.assert_not_called();self.assertFalse((Path(d)/'new/state.json').exists())
 def test_real_campaign_without_lineage_fails_before_bootstrap(self):
  from unittest.mock import Mock
  from engine import Engine
  from provider import FixtureProvider
  import sys
  sys.path.insert(0,str(LOCAL/'tests'))
  from helpers import config,reply
  with tempfile.TemporaryDirectory() as d:
   adapter=Mock();adapter.simulation_only=False;cfg=config();cfg['generalization']['mode']='counterexample_cross_scene';e=Engine(Path(d)/'new',cfg,FixtureProvider(reply),adapter)
   with self.assertRaises(AgentError):e.initialize('synthetic_guard')
   adapter.bootstrap.assert_not_called();self.assertFalse((Path(d)/'new/state.json').exists())

class LegacyPoolQuotaTests(unittest.TestCase):
 def test_missing_legacy_pool_counter_needs_explicit_host_attestation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);old={'usage':{'charged_tokens':1},'rounds_started':1,'started_epoch':0.};cfg={'budgets':{'max_rounds':12}};write(p/'state',old);write(p/'config',cfg);proof={'schema':'host_budget_lineage_v1','previous_state':asset(p/'state'),'previous_config':asset(p/'config')};write(p/'proof',proof)
   with self.assertRaises(AgentError):inherit({},dict(cfg,budget_lineage=asset(p/'proof')))
   proof['prior_validation_pool_count']=2;write(p/'proof',proof);s={};inherit(s,dict(cfg,budget_lineage=asset(p/'proof')));self.assertEqual(s['legacy_validation_pool_count'],2)
