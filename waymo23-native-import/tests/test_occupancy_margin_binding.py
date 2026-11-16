"""Occupancy expansion must use the shared reported margin, not a literal."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from scene_reliability import augment_occupancy

class MarginBindingTests(unittest.TestCase):
    def report(self,margin):
        return dict(config=dict(obstacle_margin_m=margin),obstacles=[dict(state='provisional_uncertain',box=[4.,0.,0.,1.,2.,1.,0.,0.,0.])])
    def test_shared_margin_changes_extent_and_default_stays_identical(self):
        masks=np.zeros((1,5,200,200));default=self.report(.5);legacy=dict(obstacles=default['obstacles'])
        np.testing.assert_array_equal(augment_occupancy(masks,default),augment_occupancy(masks,legacy))
        narrow=augment_occupancy(masks,self.report(.2));wide=augment_occupancy(masks,self.report(.8))
        self.assertTrue(np.all(narrow<=wide));self.assertGreater(np.count_nonzero(wide),np.count_nonzero(narrow))
        self.assertFalse(masks.any())
    def test_invalid_margins_rejected(self):
        for margin in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):augment_occupancy(np.zeros((1,5,200,200)),self.report(margin))
