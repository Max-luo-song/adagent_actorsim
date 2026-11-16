import unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path[:0]=[str(B/'waymo23-native-import'),str(B/'waymo23-native-import/waymo172_closed_loop_v1')]
from closed172_common import optimize
class VerifiedWarmTests(unittest.TestCase):
 def setUp(self):
  self.info=dict(model_lidar_frame=dict(virtual_to_ego=np.eye(4)),ego_velo=0.,ego_steer=0.)
 def test_safe_warm_is_freshly_verified_without_optimizer(self):
  masks=np.zeros((1,5,200,200));masks[:,:,180,180]=1
  with patch('closed172_common.minimize',side_effect=AssertionError('should use full verification, not solve')):
   result=optimize(np.zeros((6,2)),masks,self.info,warm=np.zeros((6,2)),reliability=dict(config={},obstacles=[]),_reuse_verified_warm=True)
  self.assertTrue(result['selected']['optimization_skipped']);self.assertGreaterEqual(result['selected']['minimum_constraint'],-1e-5)
 def test_new_obstacle_invalidates_warm_and_requires_optimizer(self):
  masks=np.zeros((1,5,200,200));masks[:,:,106,100]=1;warm=np.zeros((6,2));warm[:,0]=2.
  with patch('closed172_common.minimize',side_effect=RuntimeError('optimization required')):
   with self.assertRaisesRegex(RuntimeError,'optimization required'):
    optimize(np.zeros((6,2)),masks,self.info,warm=warm,reliability=dict(config={},obstacles=[]),_reuse_verified_warm=True)
if __name__=='__main__':unittest.main()
