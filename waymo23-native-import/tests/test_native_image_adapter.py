from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_native_image_sequence import virtual_geometry,A
class NativeImageAdapterTests(unittest.TestCase):
    def test_projection_is_independent_of_world_rigid_pose(self):
        rng=np.random.default_rng(923)
        for _ in range(10):
            relative=np.eye(4);relative[:3,:3]=Rotation.random(random_state=rng).as_matrix();relative[:3,3]=rng.normal(size=3)*10
            camera=np.eye(4);camera[:3,:3]=Rotation.random(random_state=rng).as_matrix();camera[:3,3]=rng.normal(size=3)
            mount=np.eye(4);mount[:3,3]=[0,.98,1.84]
            pose,l2c=virtual_geometry(relative,camera,mount)
            expected=np.linalg.inv(relative@camera)@np.linalg.inv(A)@pose
            np.testing.assert_allclose(l2c,expected,atol=1e-12)
