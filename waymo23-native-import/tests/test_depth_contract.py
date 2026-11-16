import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from depth_contract import expected_depth,ACCUMULATED,EXPECTED
class DepthContractTests(unittest.TestCase):
 def test_accumulated_depth_cannot_foreshorten_same_surface(self):
  alpha=np.array([[1.,.5,.25,0.]],np.float32);accumulated=alpha*10
  depth=expected_depth(accumulated,alpha,ACCUMULATED)
  np.testing.assert_allclose(depth,[[10.,10.,10.,0.]])
 def test_expected_input_is_not_normalized_twice_and_inputs_unchanged(self):
  depth=np.full((2,2),10.,np.float32);alpha=np.full((2,2),.5,np.float32);out=expected_depth(depth,alpha,EXPECTED)
  np.testing.assert_array_equal(out,depth);self.assertIsNot(out,depth)
 def test_unknown_convention_and_invalid_opacity_fail_explicitly(self):
  with self.assertRaises(ValueError):expected_depth(np.ones((1,1)),np.ones((1,1)),'RGB+ED+S')
  with self.assertRaises(ValueError):expected_depth(np.ones((1,1)),np.array([[1.5]]),ACCUMULATED)
if __name__=='__main__':unittest.main()
