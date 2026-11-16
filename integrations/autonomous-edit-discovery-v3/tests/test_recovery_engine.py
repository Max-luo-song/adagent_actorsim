
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
from common_flow import AGENT
sys.path.insert(0,str(AGENT/'tests'))
from helpers import FakeAdapter,config,reply
from provider import FixtureProvider
from recovery import RecoveryEngine

class RecoveryEngineTests(unittest.TestCase):
 def test_real_engine_can_repair_a_failed_baseline_without_relabeling_it(self):
  class Adapter(FakeAdapter):
   def bootstrap(self,c):
    x=super().bootstrap(c);x['metrics']['cutin551'].update(status='terminated',controls=116,observations=117,termination_reason='no_jointly_verified_action',error_stage=None)
    return x
   def registered_pool(self,c,champ):return {}
   def acceptance(self,candidate,baseline,reference,policy,strict):
    return dict(accepted=True,reasons=[],target_hd_gain=None,target_hd_gain_vs_v13=None,global_patch_validated=False)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=config();c['budgets']['max_rounds']=1;c['convergence']['allow_control_failure_baseline']=True
   e=RecoveryEngine(root/'campaign',c,FixtureProvider(reply),Adapter(root,[.27]));initial=e.initialize('recovery')
   self.assertEqual(initial['champion']['metrics']['cutin551']['status'],'terminated');self.assertNotIn('registry',initial)
   s=e.run();self.assertTrue(s['simulation_only']);self.assertEqual(s['history'][0]['status'],'accepted')
   self.assertFalse(s['task_convergence'][s['active_task_id']]['recovery_baseline']);self.assertEqual(s['task_convergence'][s['active_task_id']]['small_gain_streak'],0)
   self.assertEqual(s['reference_metrics']['cutin551']['status'],'completed');self.assertEqual(s['champion']['profile'],'v15_joint');self.assertTrue(s['registry'])
   self.assertFalse((root/'campaign/accepted_patch.json').exists())
