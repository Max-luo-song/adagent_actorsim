
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

import ast,sys,unittest
pass  # Path is provided by the portable boundary above
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_contract import rewrite,attach
class LiveContractTests(unittest.TestCase):
 def test_exact_frozen_driver_anchors(self):
  p=Path(__file__).resolve().parents[2]/'light-execution-horizon-v17-r2/runtime.py'
  original=ast.parse(p.read_text());f=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
  t=rewrite(ast.unparse(f));text=ast.unparse(t)
  self.assertEqual(text.count('attach_box_observation(perception, stepdir, info, spec.vehicle)'),1)
  self.assertEqual(text.count('road_reference.require_box_observation_contract = True'),1)
  # Removing only the declared attachment/assertion restores the original runtime AST.
  class Remove(ast.NodeTransformer):
   def visit_Assign(self,n):
    if ast.unparse(n).startswith(('perception = attach_box_observation(','road_reference.require_box_observation_contract =')):return None
    return n
   def visit_Assert(self,n):
    if 'observation contract unsupported' in ast.unparse(n):return None
    return n
  repaired=Remove().visit(t)
  self.assertEqual(ast.dump(repaired,include_attributes=False),ast.dump(ast.parse(ast.unparse(f)),include_attributes=False))
 def test_changed_driver_rejected(self):
  with self.assertRaises(ValueError):rewrite('def run_arm():\n pass')
 def test_missing_sidecar_is_not_clear_space(self):
  with self.assertRaises(FileNotFoundError):attach({},'/nonexistent-box-contract-test',{},None)
if __name__=='__main__':unittest.main()
