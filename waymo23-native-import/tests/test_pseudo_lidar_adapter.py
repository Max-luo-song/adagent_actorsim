import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pseudo_lidar_adapter import refine_candidates, bind_frame

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.boxes=np.array([[0.,6.,0.,2.,4.,1.5,.1,.2,.3], [1.,9.,0.,2.,4.,1.5,.2,0.,0.]])
        self.obs={'depth':{'CAM_FRONT':np.ones((10,10))}}
        self.info={'depth_encoding':'expected_camera_z_v1','cam_params':{'CAM_FRONT':{'l2c':np.eye(4)}}}
    def test_missing_depth_preserves_all(self):
        out,audit=refine_candidates({}, {}, self.boxes, [.8,.9],[0,0])
        np.testing.assert_array_equal(out,self.boxes)
        self.assertEqual(audit['fallback_reason'],'missing_depth')
    def test_all_candidates_gate_and_only_center_changes(self):
        original=self.boxes.copy()
        def fake(obs, info, box, plane, fit_yaw):
            self.assertFalse(fit_yaw)
            out=box.copy();out[1]+=.7
            return dict(accepted=True,box=out.tolist())
        with patch('depth_surface_geometry.ground_plane',return_value=np.zeros(3)), patch('depth_surface_geometry.refine',side_effect=fake) as fit:
            out,audit=refine_candidates(self.obs,self.info,self.boxes,[.8,.3],[0,0])
        self.assertEqual(fit.call_count,1)
        self.assertEqual(audit['accepted_count'],1)
        np.testing.assert_array_equal(out[:,2:],original[:,2:])
        np.testing.assert_array_equal(out[1],original[1])
        np.testing.assert_array_equal(self.boxes,original)
    def test_bad_proposal_cannot_change_yaw(self):
        box=self.boxes[0].copy();box[6]+=1
        with patch('depth_surface_geometry.ground_plane',return_value=np.zeros(3)), patch('depth_surface_geometry.refine',return_value=dict(accepted=True,box=box.tolist())):
            out,audit=refine_candidates(self.obs,self.info,self.boxes,[.8,.3],[0,0])
        np.testing.assert_array_equal(out,self.boxes)
        self.assertFalse(audit['candidates'][0]['accepted'])
    def test_frame_context_is_replaced(self):
        class Core:pass
        core=Core();bind_frame(core,{}, {'scene':'one'});old=core.geometry_candidate_refiner
        bind_frame(core,{}, {'scene':'two'})
        self.assertIsNot(core.geometry_candidate_refiner,old)
    def test_invalid_calibration_falls_back(self):
        out,audit=refine_candidates(self.obs,{},self.boxes,[.8,.9],[0,0])
        np.testing.assert_array_equal(out,self.boxes)
        self.assertTrue(audit['fallback_reason'].startswith('invalid_depth_calibration'))
