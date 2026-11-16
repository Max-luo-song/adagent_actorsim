from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edit_observability import projected_extent
class EditObservabilityTests(unittest.TestCase):
    def test_full_partial_and_no_frustum_support_are_distinct(self):
        k=np.array([[100,0,100],[0,100,100],[0,0,1]])
        corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (5.,6.)])
        result=projected_extent(corners,np.eye(4),k,(200,200))
        self.assertAlmostEqual(result['in_image_projected_fraction'],1.)
        self.assertFalse(result['truncated'])
        shifted=corners.copy();shifted[:,0]+=5.
        result=projected_extent(shifted,np.eye(4),k,(200,200))
        self.assertGreater(result['in_image_projected_fraction'],0)
        self.assertLess(result['in_image_projected_fraction'],1)
        self.assertTrue(result['truncated'])
        shifted[:,0]+=100
        self.assertEqual(projected_extent(shifted,np.eye(4),k,(200,200))['in_image_projected_fraction'],0)
    def test_behind_camera_does_not_invent_visibility(self):
        corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-2.,-1.)])
        result=projected_extent(corners,np.eye(4),np.eye(3),(200,200))
        self.assertFalse(result['projectable'])
        self.assertFalse(result['occlusion_verified'])
