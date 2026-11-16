
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

import ast
import copy
pass  # Path is provided by the portable boundary above
import tempfile
import unittest
from unittest.mock import patch
from common_flow import challenge,EDIT_GRID,normalize_contract,write,read,sha,asset,verify,healthy


def metric(score=.25):return dict(status='completed',controls=188,observations=189,collision=False,map_departure=False,unresolved_stop=False,score=dict(hdscore=score,pdms=1.,rc=score))

class DiscoveryFlowTests(unittest.TestCase):
 def test_new_edit_grid_does_not_reuse_canonical_parameters(self):
  self.assertTrue(all((v['lane_change_duration_s'],v['actor_deceleration_duration_s'])!=(2.8,4.8) for v in EDIT_GRID))
  self.assertEqual(len({v['id'] for v in EDIT_GRID}),len(EDIT_GRID))
 def test_complete_repeated_gap_with_executed_alternative_qualifies(self):
  self.assertTrue(challenge(metric(),metric(),metric(.27),metric(.28))['qualified'])
 def test_no_gain_does_not_invent_a_failure(self):
  self.assertFalse(challenge(metric(),metric(),metric(),metric(.28))['qualified'])
 def test_failed_original_is_not_new_edit_failure(self):
  bad=metric(.28);bad['status']='terminated'
  self.assertFalse(challenge(metric(),metric(),metric(.27),bad)['qualified'])
 def test_unreproduced_problem_is_not_qualified(self):
  self.assertFalse(challenge(metric(),metric(.26),metric(.27),metric(.28))['qualified'])
 def test_witness_collision_not_feasible(self):
  bad=metric(.27);bad['collision']=True
  self.assertFalse(challenge(metric(),metric(),bad,metric(.28))['qualified'])
 def test_path_receipts_normalize_by_content(self):
  self.assertEqual(normalize_contract(dict(x=dict(path='/a',sha256='x'))),normalize_contract(dict(x=dict(path='/b',sha256='x'))))
 def test_changed_manifest_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'task.json';write(p,{'x':1});receipt=asset(p);write(p,{'x':2})
   with self.assertRaises(ValueError):verify(receipt)
 def test_host_syntax(self):
  for p in Path(__file__).resolve().parents[1].glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 def test_handoff_runs_v7_and_carries_all_existing_usage(self):
  import sys,flow
  sys.path.insert(0,str(flow.AGENT))
  import engine,task_adapter
  calls={}
  class Adapter:
   def __init__(self,task):calls['task']=task
  class Engine:
   def __init__(self,root,config=None,adapter=None):calls['root']=root;calls['config']=config
   def initialize(self,name):calls['campaign']=name;return {}
   def save(self,state):calls['state']=copy.deepcopy(state)
   def run(self):calls['run']=True;return dict(status='waiting_api',active_task_id='new_task')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);taskfile=root/'task.json';write(taskfile,dict(id='new_task',host_pins={}))
   old=dict(usage=dict(api_requests=1,charged_tokens=34660,reported_tokens=34660,gpu_attempts=1,gpu_dispatches=1),rounds_started=2,history=[])
   write(root/'source_state.json',old);write(root/'source_config.json',dict(budgets=dict(max_gpu_attempts=6),presentation=dict(enabled=True)));write(root/'ledger.json',dict(gpu_attempts=3))
   m=dict(host_pins={},explicit_budget_extension=dict(gpu_attempts=16,wall_seconds=14400))
   with patch.object(flow,'OUT',root),patch.object(flow,'P26',root/'p26'),patch.object(flow,'remaining',return_value=100),patch.object(engine,'Engine',Engine),patch.object(task_adapter,'NewEditAdapter',Adapter):
    flow.handoff(m,taskfile)
   self.assertTrue(calls['run']);self.assertEqual(calls['state']['usage']['charged_tokens'],34660);self.assertEqual(calls['state']['usage']['gpu_attempts'],4);self.assertEqual(calls['state']['rounds_started'],2);self.assertEqual(calls['config']['budgets']['max_gpu_attempts'],22);self.assertFalse(calls['state']['budget_lineage']['counters_reset'])

if __name__=='__main__':unittest.main()
