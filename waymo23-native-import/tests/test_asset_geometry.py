from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_geometry import local_corners,source_corners,bottom_polygon,declared_contract,validate_pose
from asset_placement import terrain_aligned_pose
from paired_edit_diagnostic import asset_corners

class AssetGeometryTests(unittest.TestCase):
    def test_scene_independent_placement_and_projection_share_axis_and_dimensions(self):
        for heading in (0,.7,-1.35):
            pose=terrain_aligned_pose([2,3,4],heading,dict(origin_xy=[0,0],plane=[.04,-.03,1]),.2)
            asset=dict(asset_wlh=[1.6,4.4,1.5],local_bottom_y=.2,placements=[dict(asset_to_source=pose.tolist())])
            expected=local_corners(asset['asset_wlh'],.2)@pose[:3,:3].T+pose[:3,3]
            np.testing.assert_allclose(asset_corners(asset),expected,atol=1e-12)
            recovered=(source_corners(asset)-pose[:3,3])@pose[:3,:3]
            np.testing.assert_allclose(np.ptp(recovered,axis=0),[4.4,1.5,1.6],atol=1e-12)
            bottom=bottom_polygon(asset['asset_wlh'],.2)
            np.testing.assert_array_equal(bottom, [[-2.2,.2,-.8],[2.2,.2,-.8],[2.2,.2,.8],[-2.2,.2,.8]])
    def test_bad_geometry_and_unsupported_convention_fail_closed(self):
        for pose in (np.diag([-1.,1.,1.,1.]),np.diag([2.,1.,1.,1.])):
            with self.assertRaises(ValueError):validate_pose(pose)
        with self.assertRaises(ValueError):local_corners([1,-2,1],0)
        with self.assertRaises(ValueError):source_corners(dict(geometry_contract=dict(schema='unknown')))
    def test_contract_does_not_promote_gaussians_to_physical_truth(self):
        c=declared_contract([1.6,4.4,1.5],.2)
        self.assertFalse(c['physical_dimensions_independently_verified'])
        self.assertFalse(c['real_wheel_contacts_verified'])
        self.assertFalse(c['signed_front_verified'])
