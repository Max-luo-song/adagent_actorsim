import sys,unittest
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path[:0]=[str(B/'integrations/sr-hugsim'),str(B/'waymo23-native-import'),str(B/'waymo23-native-import/waymo172_closed_loop_v1')]
from native_geometry_evaluation import NativeVehicleEvaluation
from runtime_configuration import load_vehicle_config
class NativeEvaluationTests(unittest.TestCase):
 def make(self,visible=True):
  return NativeVehicleEvaluation(dict(instances_trans=np.array([[[4.,0.,0.]]]),instances_quats=np.array([[[1.,0.,0.,0.]]]),instances_size=np.array([[4.,2.,1.6]]),instances_fv=np.array([[visible]])))
 def test_separated_initial_vehicle_and_later_overlap(self):
  evaluator=self.make();pose=np.eye(4);r=evaluator.check(pose,0,load_vehicle_config());self.assertFalse(r['overlap']);self.assertAlmostEqual(r['nearest']['gap_m'],.5)
  pose[0,3]=2.;self.assertTrue(evaluator.check(pose,0,load_vehicle_config())['overlap'])
 def test_inactive_native_object_is_not_a_collision(self):
  pose=np.eye(4);pose[0,3]=4.;self.assertFalse(self.make(False).check(pose,0,load_vehicle_config())['overlap'])
 def test_world_transform_preserves_evaluation(self):
  evaluator=self.make();pose=np.eye(4);expected=evaluator.check(pose,0,load_vehicle_config())['nearest']['gap_m'];evaluator.trans+=np.array([10.,-7.,0.]);pose[:2,3]=[10.,-7.];self.assertAlmostEqual(evaluator.check(pose,0,load_vehicle_config())['nearest']['gap_m'],expected)
