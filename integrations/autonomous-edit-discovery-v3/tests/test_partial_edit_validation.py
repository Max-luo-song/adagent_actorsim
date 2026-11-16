
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
from common_flow import write,read,asset
from scene_designer import Designer,validate_batch,compile_entries
from core import digest,AgentError
from provider import FixtureProvider
from test_scene_designer import context,value,config

class PartialEditTests(unittest.TestCase):
 def test_invalid_neighbor_keeps_original_valid_edit_and_reports_phase(self):
  c=context();v=value(c);good=copy.deepcopy(v['edits'][0]);bad=copy.deepcopy(good);bad['parameters']['phases'][1]['action']='stop_hold'
  v['edits']=[bad,good];before=copy.deepcopy(v)
  edits,selection=validate_batch(v,digest(c),[])
  self.assertEqual(len(edits),1);self.assertEqual(selection['accepted_edit_indices'],[2]);self.assertEqual(selection['rejected_edits'][0]['edit_index'],1);self.assertEqual(selection['rejected_edits'][0]['phase_index'],2);self.assertEqual(v,before)
 def test_duplicate_only_rejects_its_own_edit(self):
  c=context();v=value(c);v['edits']*=2
  edits,selection=validate_batch(v,digest(c),[]);self.assertEqual(len(edits),1);self.assertEqual(selection['accepted_edit_indices'],[1])
 def test_context_or_envelope_corruption_still_rejects_all(self):
  c=context();v=value(c);v['context_sha256']='bad'
  with self.assertRaises(AgentError):validate_batch(v,digest(c),[])
 def test_cached_real_response_recovers_valid_subset_without_API_and_preserves_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);c=context();v=value(c);bad=copy.deepcopy(v['edits'][0]);bad['parameters']['phases'][0]['target_speed_ratio']=.8;v['edits'].insert(0,bad)
   record=root/'batch_001/request_001.json';write(record,dict(content=json.dumps(v),metadata=dict(external_API_called=True,finish_reason='stop',usage_known=False)))
   receipt=asset(record)
   class Missing(FixtureProvider):
    def readiness(self):return ['PATCH_API_KEY']
   provider=Missing(lambda *a:None);model=Designer(root,config(),{},provider);proposal=model.propose(c)
   self.assertEqual(provider.calls,0);self.assertEqual(asset(record),receipt);self.assertEqual(proposal['selection']['accepted_edit_indices'],[2])
   entries=compile_entries(proposal,1);self.assertIn('_e02_',entries[0]['id']);self.assertEqual(entries[0]['phases'],v['edits'][1]['parameters']['phases'])
 def test_model_receives_cross_phase_rules_and_exact_failure_feedback(self):
  c=context()
  def reply(messages,n):
   payload=_hugsim_expand(json.loads(messages[-1]['content']));self.assertIn('stop_hold',payload['action_semantics']['rules'])
   v=value(c)
   if n==1:v['edits'][0]['parameters']['phases'][0]['target_speed_ratio']=.8
   else:
    failure=payload['validation_feedback'][0]['rejected_edits'][0];self.assertEqual(failure['edit_index'],1);self.assertEqual(failure['phase_index'],1)
   return v
  with tempfile.TemporaryDirectory() as d:
   provider=FixtureProvider(reply);Designer(d,config(),{},provider).propose(c);self.assertEqual(provider.calls,2)
