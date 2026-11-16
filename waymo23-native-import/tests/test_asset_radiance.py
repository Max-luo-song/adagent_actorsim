from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_radiance import sh_rotation,basis

class AssetRadianceTests(unittest.TestCase):
    def test_rotated_coefficients_preserve_local_view_colors(self):
        rng=np.random.default_rng(2351)
        coefficients=rng.normal(size=(16,3))
        directions=rng.normal(size=(100,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        for r in Rotation.random(12,random_state=rng).as_matrix():
            m=sh_rotation(r)
            np.testing.assert_allclose(basis(directions,3)@m@coefficients,basis(directions@r,3)@coefficients,atol=1e-12)
            np.testing.assert_allclose(m.T@m,np.eye(16),atol=1e-12)
        # Negative control: unchanged coefficients are not rotation-covariant.
        self.assertGreater(np.max(np.abs(basis(directions,3)@coefficients-basis(directions@r,3)@coefficients)),.1)

    def test_composition_identity_and_degree_zero(self):
        a,b=Rotation.from_euler('xyz',[[.2,.4,-.7],[.8,-.6,.3]]).as_matrix()
        np.testing.assert_allclose(sh_rotation(a@b),sh_rotation(a)@sh_rotation(b),atol=1e-12)
        np.testing.assert_allclose(sh_rotation(np.eye(3)),np.eye(16),atol=1e-12)
        np.testing.assert_array_equal(sh_rotation(a,0),[[1.]])

    def test_improper_rotation_rejected(self):
        for r in (np.diag([1,1,-1]),np.eye(3)*2,np.full((3,3),np.nan)):
            with self.assertRaises(ValueError):sh_rotation(r)
