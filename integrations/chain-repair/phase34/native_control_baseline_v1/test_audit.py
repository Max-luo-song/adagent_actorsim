
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
import unittest,json,pickle
from cases import load
from constraint_audit import initial_components

class ConstraintAuditTests(unittest.TestCase):
 def test_old_conflict_identifies_depth_not_neural_raster(self):
  _,rel,info,plan,occ,v,states,times=load(15);before=pickle.dumps((rel,info,occ),protocol=4)
  audit=initial_components(occ,rel,info,v)
  self.assertGreater(audit['network_raster_gap_m'],13.)
  self.assertLess(audit['observed_depth_gap_m'],0.)
  self.assertEqual(audit['blocking_sources'],['observed_depth_gap_m'])
  self.assertEqual(before,pickle.dumps((rel,info,occ),protocol=4))
 def test_repaired_geometry_has_separate_positive_gaps(self):
  _,rel,info,plan,occ,v,states,times=load(15);rel=_hugsim_expand(json.loads((HERE/'validation_runs_v2/guard_input_015.json').read_text()))
  audit=initial_components(occ,rel,info,v)
  self.assertGreater(audit['observed_depth_gap_m'],2.)
  self.assertEqual(audit['blocking_sources'],[])
 def test_empty_unknown_geometry_is_serializable_without_infinity(self):
  _,rel,info,plan,occ,v,states,times=load(0)
  audit=initial_components(occ,rel,info,v)
  self.assertIsNone(audit['unknown_boundary_gap_m']);json.dumps(audit,allow_nan=False)

if __name__=='__main__':unittest.main(verbosity=2)
