import copy
import unittest
import numpy as np
from window_contract import validate_manifest,CAMERAS

class WindowTests(unittest.TestCase):
    def setUp(self):
        self.data=dict(frames=[80,81,82],camera_order=CAMERAS,scene_edits=[],
            mask_valid_for_novel_or_edited_poses=False,records=[dict(frame=f,camera_id=c,camera=n,
            timestamp_s=f/10,ego_to_world=np.eye(4).tolist()) for f in (80,81,82) for c,n in enumerate(CAMERAS)])
    def test_frame_major_batch(self): self.assertEqual(len(validate_manifest(self.data)),9)
    def test_swapped_camera_rejected(self):
        self.data['records'][1],self.data['records'][2]=self.data['records'][2],self.data['records'][1]
        with self.assertRaises(ValueError):validate_manifest(self.data)
    def test_stale_mask_pose_rejected(self):
        self.data['records'][1]['ego_to_world'][0][3]=1
        with self.assertRaises(ValueError):validate_manifest(self.data)
    def test_time_mismatch_rejected(self):
        self.data['records'][4]['timestamp_s']+=.1
        with self.assertRaises(ValueError):validate_manifest(self.data)
    def test_edited_recorded_masks_rejected(self):
        self.data['scene_edits']=['move pedestrian']
        with self.assertRaises(ValueError):validate_manifest(self.data)

if __name__=='__main__':unittest.main()
