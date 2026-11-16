import unittest
import numpy as np
from shape_metrics import shape_residual

class ShapeTests(unittest.TestCase):
    def test_rigid_motion_is_not_body_deformation(self):
        a = np.random.default_rng(7).normal(size=(100, 3))
        r = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        self.assertLess(shape_residual(a, a @ r + [5,2,8])['rms_m'], 1e-12)
    def test_articulation_survives_alignment(self):
        a = np.random.default_rng(7).normal(size=(100, 3))
        b = a.copy(); b[:40, 2] += 0.2
        self.assertGreater(shape_residual(a,b)['rms_m'], 0.05)
    def test_reflection_is_not_rotation(self):
        a = np.random.default_rng(7).normal(size=(100,3))
        self.assertGreater(shape_residual(a,a * [-1,1,1])['rms_m'], .1)
    def test_bad_data_rejected(self):
        with self.assertRaises(ValueError): shape_residual([[0,0,0]],[[0,0,0]])

if __name__ == '__main__': unittest.main()
