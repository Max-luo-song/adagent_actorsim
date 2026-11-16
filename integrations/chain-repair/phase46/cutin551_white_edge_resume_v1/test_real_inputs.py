
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

import copy,pickle,sys,unittest
pass  # Path is provided by the portable boundary above
from paths import *
sys.path.insert(0,str(DRIVER))
from bootstrap import *
from resume_store import canonical,require_same
from temporal_context import AgentlessForecast
class RealInputTests(unittest.TestCase):
 def test_real_saved_optimizer_contracts_and_agentless_are_serializable(self):
  files=sorted((CR/'phase44/cutin551_optimized_v1/validation/inputs').glob('step_*.pkl'));self.assertEqual(len(files),28)
  for file in files:
   with file.open('rb') as f:data=pickle.load(f)
   payload={k:data[k] for k in ('reference','occupancy','info','kwargs')};scene=payload['info']['scene_token'];a=canonical(payload,Path('/old'),scene);b=canonical(copy.deepcopy(payload),Path('/old'),scene);self.assertEqual(a,b,file.name)
  a=canonical(AgentlessForecast(),Path('/old'),'a:1');b=canonical(AgentlessForecast(),Path('/new'),'b:1');self.assertEqual(a,b)
if __name__=='__main__':unittest.main(verbosity=2)
