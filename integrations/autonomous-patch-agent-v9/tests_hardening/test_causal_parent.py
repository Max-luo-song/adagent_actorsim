"""Causal software tests: unchanged v8 fails assertions, v9 passes; synthetic only."""

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

import copy,importlib.util,sys,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];sys.path[:0]=[str(LOCAL),str(LOCAL/'tests')]
from helpers import metrics
import governance
from core import AgentError

def parent():
 file=LOCAL.parent/'autonomous-patch-agent-v8/governance.py';spec=importlib.util.spec_from_file_location('_unchanged_v8_governance',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sample(module):
 b=metrics(.26);p={'pool_id':'seed','profile':'v15_joint','version':'seed','reference_metrics':b};pool={'seed':dict(p,eligible=True,metrics=b,evaluated_repairs=0)};bundle={'round_id':'unqualified','code':'synthetic','release':'synthetic','release_name':'synthetic','candidate_pins':{}}
 n=metrics(.9);n['original551']['score']['rc']=.29;module.add_result(pool,p,bundle,'synthetic',n);return pool
class CausalTests(unittest.TestCase):
 def test_unchanged_parent_fails_anchor_contamination_assertion(self):
  module=parent()
  class ParentAssertion(unittest.TestCase):
   def runTest(self):self.assertFalse(sample(module)['unqualified']['eligible'])
  result=unittest.TestResult();ParentAssertion().run(result);self.assertEqual(len(result.failures),1);self.assertEqual(len(result.errors),0)
 def test_v9_passes_identical_assertion(self):self.assertFalse(sample(governance)['unqualified']['eligible'])
