import unittest
import numpy as np
from uniad_analysis import match,region_occupancy


class AnalysisTests(unittest.TestCase):
    def test_true_car_vs_wrong_class_and_offset(self):
        truth=np.array([[-1.,-2.],[-1.,2.],[1.,2.],[1.,-2.]])
        data=dict(boxes=np.array([[0.,0.,0.,2.,4.,1.5,0.]]),scores=np.array([.6]),labels=np.array([0]))
        result=match(data,truth)
        self.assertTrue(result['matched']);self.assertAlmostEqual(result['iou'],1.)
        self.assertAlmostEqual(result['center_error_m'],0.);self.assertAlmostEqual(result['heading_error_mod180_deg'],0.)
        data['labels'][0]=1;self.assertFalse(match(data,truth)['matched'])
        data['labels'][0]=0;data['boxes'][0,0]=5
        self.assertFalse(match(data,truth)['matched'])
    def test_local_occupancy_excludes_distant_cells_and_preserves_horizons(self):
        masks=np.zeros((1,5,1,200,200));masks[0,2,0,100,100]=1;masks[0,0,0,150,150]=1
        truth=np.array([[-1.,-2.],[-1.,2.],[1.,2.],[1.,-2.]])
        self.assertEqual(region_occupancy(masks,truth),[0,0,1,0,0])

if __name__=='__main__':unittest.main()
