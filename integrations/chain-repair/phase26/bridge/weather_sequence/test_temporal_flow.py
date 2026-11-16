import unittest
import numpy as np
from apply_weather_window import flow_metrics
try:
    import cv2
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'OpenCV shared libraries unavailable on this host')
class FlowTests(unittest.TestCase):
    def setUp(self):
        self.a=np.random.default_rng(42).integers(20,200,(64,96,3),dtype=np.uint8)
    def test_unchanged_frames_and_weather_have_zero_flicker(self):
        w=self.a+20
        r=flow_metrics(self.a,self.a,w,w)
        self.assertGreater(r['valid_fraction'],.8)
        self.assertLess(r['clear_aligned_mae'],.01)
        self.assertLess(r['weather_aligned_mae'],.01)
    def test_temporal_exposure_change_is_detected(self):
        r=flow_metrics(self.a,self.a,self.a,self.a+20)
        self.assertAlmostEqual(r['weather_aligned_mae'],20,places=1)

if __name__=='__main__':unittest.main()
