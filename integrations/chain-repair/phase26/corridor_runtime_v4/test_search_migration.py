from bootstrap import *
import shutil,tempfile,unittest
from change_contract import validate,PRESERVED,CORE
class PolicyContractTests(unittest.TestCase):
 def fixture(self,root):
  ref=root/'motion_search_v3';p=root/'candidate';ref.mkdir();p.mkdir()
  for name in set(PRESERVED)|{'empty_controller.py'}:shutil.copyfile(HERE.parent/'motion_search_v3'/name,ref/name)
  for name in set(PRESERVED)|set(CORE)|{'change_receipt.json'}:shutil.copyfile(HERE/name,p/name)
  return p
 def test_actual_base_physics_and_evidence_are_preserved(self):self.assertTrue(validate(HERE))
 def test_white_car_and_memory_fixes_cannot_be_replaced(self):
  for name in ['time_local_motion.py','motion_memory.py']:
   with tempfile.TemporaryDirectory() as d:
    p=self.fixture(Path(d));(p/name).write_text('# invalid replacement')
    with self.assertRaises(ValueError):validate(p)
 def test_physical_integrator_cannot_change(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(Path(d));f=p/'empty_controller.py';f.write_text(f.read_text().replace('speed += acc * dt','speed += acc * dt * 2'))
   with self.assertRaises(ValueError):validate(p)
 def test_unverified_cost_or_corridor_change_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(Path(d));(p/'road_priority.py').write_text('# changed cost')
   with self.assertRaises(ValueError):validate(p)
if __name__=='__main__':unittest.main()
