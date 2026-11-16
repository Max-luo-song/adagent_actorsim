
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
import json,shutil,tempfile,unittest
from migration_contract import validate,PRESERVED
class MigrationContractTests(unittest.TestCase):
 def fixture(self,root):
  original=root/'motion_envelope_v2';candidate=root/'candidate';original.mkdir();candidate.mkdir()
  reference=HERE.parent/'motion_envelope_v2'
  for name in set(PRESERVED)|{'empty_controller.py','consensus_search.py','runtime_policy.json'}:shutil.copyfile(reference/name,original/name)
  for name in set(PRESERVED)|{'empty_controller.py','consensus_search.py','runtime_policy.json','baseline_consensus.py','test_consensus_search.py','migration_receipt.json'}:shutil.copyfile(HERE/name,candidate/name)
  return candidate
 def test_actual_source_preserves_own_repairs(self):self.assertTrue(validate(HERE))
 def test_white_car_or_road_repair_cannot_be_silently_replaced(self):
  for name in ['time_local_motion.py','motion_memory.py','native_objective.py']:
   with tempfile.TemporaryDirectory() as d:
    candidate=self.fixture(Path(d));(candidate/name).write_text('# replaced upstream component\n')
    with self.assertRaisesRegex(ValueError,'Own fix'):validate(candidate)
 def test_extra_controller_changes_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   candidate=self.fixture(Path(d));f=candidate/'empty_controller.py';f.write_text(f.read_text().replace('maxiter=220','maxiter=20'))
   with self.assertRaisesRegex(ValueError,'Controller changed'):validate(candidate)
 def test_scaled_solver_cannot_be_enabled_by_migration(self):
  with tempfile.TemporaryDirectory() as d:
   candidate=self.fixture(Path(d));f=candidate/'runtime_policy.json';policy=_hugsim_expand(json.loads(f.read_text()));policy['scaled_slsqp']=True;f.write_text(json.dumps(policy))
   with self.assertRaisesRegex(ValueError,'runtime policy'):validate(candidate)
if __name__=='__main__':unittest.main(verbosity=2)
