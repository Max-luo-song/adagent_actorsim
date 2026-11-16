from pathlib import Path
import sys
import json
import pickle
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_edit_diagnostic import image_change_audit


class PairedImageAuditTests(unittest.TestCase):
    def test_local_image_change_does_not_claim_independent_reference_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for role in ('edited', 'baseline'):
                folder = root/role
                folder.mkdir()
                rgb = np.zeros((20, 20, 3), dtype=np.uint8)
                if role == 'edited':
                    rgb[10, 10] = 255
                packet = folder/'observation.pkl'
                with packet.open('wb') as stream:
                    pickle.dump((dict(rgb=dict(CAM_FRONT=rgb)), {}), stream)
                sequence = root/(role+'.json')
                sequence.write_text(json.dumps(dict(frames=[dict(frame=0, packet=str(packet))])))
                paths[role] = sequence
            pose = np.diag([1., -1., -1., 1.])
            pose[:3, 3] = [0, 0, 8]
            np.savez(root/'edited/render_evidence.npz',
                     CAM_FRONT__K=np.array([[10., 0, 10], [0, 10., 10], [0, 0, 1]]),
                     CAM_FRONT__camera_to_world=pose, CAM_FRONT__alpha=np.ones((20, 20)),
                     CAM_FRONT__depth=np.full((20, 20), 8.))
            asset_pose = np.eye(4)
            asset_pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
            asset = root/'asset.json'
            asset.write_text(json.dumps(dict(asset_wlh=[1., 2., 1.], local_bottom_y=0.,
                                             placements=[dict(asset_to_source=asset_pose.tolist())])))
            report = image_change_audit(paths['edited'], paths['baseline'], asset)
            row = report['records'][0]
            self.assertGreater(row['changed_fraction_inside_halo'], 0.)
            self.assertEqual(row['changed_fraction_outside_halo'], 0.)
            self.assertFalse(report['independent_reference_compared'])
            self.assertFalse(report['n1_edit_certified'])
