import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from run_asset_quality import supported_view_pose,camera_azimuth

class ViewTests(unittest.TestCase):
    def test_supported_view_changes_only_heading_and_hits_measured_azimuth(self):
        pose=np.eye(4);pose[:3,:3]=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
        pose[:3,3]=[10.,2.,1.];before=pose.copy();camera=np.array([0.,0.,2.])
        corrected,yaw=supported_view_pose(pose,camera,-57.1855)
        self.assertAlmostEqual(np.rad2deg(camera_azimuth(corrected,camera)),-57.1855,places=4)
        np.testing.assert_array_equal(corrected[:3,3],pose[:3,3])
        np.testing.assert_array_equal(pose,before)
        np.testing.assert_allclose(corrected[:3,:3].T@corrected[:3,:3],np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(corrected[:3,:3]),1.)

if __name__=='__main__':unittest.main()
