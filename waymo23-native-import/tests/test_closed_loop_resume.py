import sys,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'integrations/sr-hugsim'))
from closed_loop_resume import save_checkpoint,load_checkpoint
class ResumeTests(unittest.TestCase):
 def test_round_trip_restores_post_control_state_and_rejects_changed_experiment(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'outputs';out.mkdir();p=out/'state.pkl';motion=SimpleNamespace(x=2.,z=3.,speed=1.,timestamp=2.)
   save_checkpoint(p,implementation={'code':'sha'},source_result=out,next_index=10,token='model-token',motion=motion,warm=np.ones((6,2)),controller={'state':'normal'},obstacle_memory={'anchor':np.array([1.,2.])},depth_memory={'cell':1},report={'experiment_signature':'fixed-scene'},minimum_gap=.8)
   r=load_checkpoint(p,{'code':'sha'},root,'fixed-scene');self.assertEqual(r['motion']['x'],2.);self.assertEqual(r['next_index'],10);self.assertEqual(r['minimum_gap'],.8);np.testing.assert_array_equal(r['warm'],np.ones((6,2)))
   with self.assertRaises(ValueError):load_checkpoint(p,{'code':'different'},root,'fixed-scene')
   with self.assertRaises(ValueError):load_checkpoint(p,{'code':'sha'},root,'edited-scene')
 def test_source_outside_output_tree_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):load_checkpoint(Path(tmp)/'foreign.pkl',{},Path(tmp),'x')
if __name__=='__main__':unittest.main()
