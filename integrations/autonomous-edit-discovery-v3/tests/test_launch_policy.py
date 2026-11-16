
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

import copy,unittest
from launch_actions import additions
from common_flow import read,AGENT
from validation import acceptance

class LaunchPolicyTests(unittest.TestCase):
 def test_expired_wall_allocation_adds_window_without_resetting_start(self):
  ledger=dict(started_epoch=100.,gpu_attempts=8);old=copy.deepcopy(ledger)
  x=additions(dict(explicit_budget_extension=dict(wall_seconds=14400)),ledger,14400,12,86500.)
  self.assertEqual(x,dict(add_wall_seconds=86400,add_gpu_attempts=12));self.assertEqual(ledger,old)
 def test_lateral_is_diagnostic_but_physical_and_score_gates_remain(self):
  def metric(score):return dict(score=dict(hdscore=score,pdms=1.,rc=score),status='completed',controls=188,observations=189,collision=False,map_departure=False,unresolved_stop=False,progress_m=100.,stopped_s=0.,final_abs_lateral_m=1.,peak_abs_lateral_m=2.)
  base={k:metric(.26) for k in ('original551','cutin551')};n=copy.deepcopy(base);n['cutin551']['score']['hdscore']=.27
  policy=read(AGENT/'config.example.json')['acceptance'];policy['lateral_diagnostics_only']=True
  n['cutin551'].update(final_abs_lateral_m=10.,peak_abs_lateral_m=11.)
  self.assertTrue(acceptance(n,base,base,policy,{})['accepted'])
  for key in ('collision','map_departure','unresolved_stop'):
   bad=copy.deepcopy(n);bad['cutin551'][key]=True;self.assertFalse(acceptance(bad,base,base,policy,{})['accepted'])
  bad=copy.deepcopy(n);bad['original551']['score']['hdscore']=.1;self.assertFalse(acceptance(bad,base,base,policy,{})['accepted'])

 def test_missing_api_is_durable_and_does_not_start_a_worker(self):
  import tempfile,sys
  pass  # Path is provided by the portable boundary above
  from unittest.mock import patch
  import launch_actions as launch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with patch.object(launch,'OUT',root),patch.object(sys,'argv',['launch_actions.py']),patch.object(launch,'HTTPProvider') as provider,patch.object(launch.subprocess,'run') as run:
    provider.return_value.readiness.return_value=['PATCH_API_KEY']
    self.assertEqual(launch.main(),2);run.assert_not_called()
   state=read(root/'startup.json');self.assertEqual(state['stage'],'waiting_api_launch');self.assertFalse(state['coordinator_started']);self.assertFalse((root/'owner.json').exists())

 def test_explicit_window_extension_preserves_attempts_tokens_and_start(self):
  import tempfile
  pass  # Path is provided by the portable boundary above
  from unittest.mock import patch
  import launch_actions as launch
  from common_flow import write
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);mission=dict(explicit_budget_extension=dict(wall_seconds=100,gpu_attempts=28),host_pins={});ledger=dict(started_epoch=100.,gpu_attempts=10);usage=dict(api_requests=4,charged_tokens=73680)
   write(root/'mission.json',mission);write(root/'ledger.json',ledger);write(root/'usage.json',usage)
   with patch.object(launch,'OUT',root),patch.object(launch.time,'time',return_value=190.):launch.extend_window(60)
   after=read(root/'mission.json');self.assertEqual(after['explicit_budget_extension'],dict(wall_seconds=150,gpu_attempts=28));self.assertEqual(read(root/'ledger.json'),ledger);self.assertEqual(read(root/'usage.json'),usage)
 def test_discovery_extension_cannot_ignore_a_frozen_patch_campaign(self):
  import tempfile
  pass  # Path is provided by the portable boundary above
  from unittest.mock import patch
  import launch_actions as launch
  from common_flow import write
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'handoff.json',{})
   with patch.object(launch,'OUT',root):
    with self.assertRaises(ValueError):launch.extend_window(14400)
