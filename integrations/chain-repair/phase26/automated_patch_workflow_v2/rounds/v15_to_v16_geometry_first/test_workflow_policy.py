
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

import importlib.util,unittest
pass  # Path is provided by the portable boundary above
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_geometry_first_workflow',D/'iterate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PolicyTests(unittest.TestCase):
 def test_existing_v15_score_gain_does_not_hide_path_regression(self):
  m.OUT=m.P/'patch_v15_validation_r2_v1';r=m.path_comparison()
  self.assertFalse(r['passed']);self.assertFalse(r['cases']['cutin551']['passed'])
  self.assertTrue(r['cases']['original551']['passed'])
 def test_identical_v13_pair_passes_lateral_gate(self):
  m.OUT=m.P/'patch_v13_validation_r2_v1';r=m.path_comparison();self.assertTrue(r['passed'])
if __name__=='__main__':unittest.main(verbosity=2)
