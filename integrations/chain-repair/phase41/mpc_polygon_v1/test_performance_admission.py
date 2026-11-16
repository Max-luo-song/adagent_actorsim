
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import copy,json,unittest
from performance_acceptance import validate_document,validate
class PerformanceAdmissionTests(unittest.TestCase):
 def data(self):return _hugsim_expand(json.loads((HERE/'performance_acceptance.json').read_text()))
 def test_actual_measured_candidate_is_admitted(self):self.assertTrue(validate(HERE))
 def test_runtime_gain_cannot_hide_quality_regression(self):
  data=self.data();data['paired_frames'][0]['objective_increase_fraction']=.1
  with self.assertRaises(ValueError):validate_document(data)
 def test_unverified_full_expansion_cannot_be_admitted(self):
  data=self.data();data['held_out_expansion']['original_starts_preserved']=False
  with self.assertRaises(ValueError):validate_document(data)
if __name__=='__main__':unittest.main(verbosity=2)
