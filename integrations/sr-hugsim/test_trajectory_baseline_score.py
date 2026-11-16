
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import copy,sys,unittest
pass  # Path is provided by the portable boundary above
import numpy as np
BASE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BASE/'waymo23-native-import'),str(BASE/'waymo23-native-import/waymo172_closed_loop_v1')]
from trajectory_baseline_score import score_report
from runtime_configuration import load_vehicle_config
from waymo_adapter import SR_TO_HUGSIM as T
from test_uniad_bridge import UniADBridgeTests
from uniad_bridge import to_uniad

class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.vehicle=load_vehicle_config()
        asset_pose=np.eye(4);asset_pose[2,3]=5
        self.asset=dict(asset_wlh=[2,4,1.5],local_bottom_y=0,placements=[dict(case='heading_A',asset_to_native=asset_pose.tolist())])
        def pose(z):
            native=np.eye(4);native[0,3]=5;native[2,3]=z
            return (T.T@native@T).tolist()
        self.report=dict(scene=172,start_frame=146,steps_requested=2,dt=.1,guard_policy='test',
            observations=[dict(timestamp=0,closed_ego_to_world=pose(0),closed_speed_mps=2,inserted_reference_gap_m=2),dict(timestamp=.1,closed_ego_to_world=pose(10),closed_speed_mps=2,inserted_reference_gap_m=2)],
            trace=[dict(action=dict(acc=0,steer_rate=0)) for _ in range(2)],minimum_inserted_reference_gap_m=2)
    def test_clear_pass_steady_controls_scores_100(self):
        result=score_report(self.report,self.asset,self.vehicle)
        self.assertEqual(result['score'],100.)
        self.assertTrue(result['metrics']['passed_asset_far_plane'])
    def test_collision_cannot_get_high_score(self):
        self.report.update(error='Interpolated reference overlap after control',minimum_inserted_reference_gap_m=0)
        self.assertEqual(score_report(self.report,self.asset,self.vehicle)['score'],0.)
    def test_incomplete_and_small_clearance_are_penalized(self):
        self.report['steps_requested']=4;self.report['minimum_inserted_reference_gap_m']=.1
        result=score_report(self.report,self.asset,self.vehicle)
        self.assertEqual(result['components']['clearance'],12.5)
        self.assertEqual(result['completion_fraction'],.5)
        self.assertFalse(result['metrics']['clearance_satisfied'])
    def test_source_frame_can_hold_without_freezing_model_clock(self):
        packet=UniADBridgeTests().packet();packet.update(source_frame=197,timestamp_s=19.7)
        _,before=to_uniad(packet,np.eye(4),speed_mps=2,steering_rad=0,command=2,scene_token='same')
        packet.update(timestamp_s=19.8,background_timestamp_s=19.7)
        _,after=to_uniad(packet,np.eye(4),speed_mps=2,steering_rad=0,command=2,scene_token='same')
        self.assertEqual(before['source_frame'],after['source_frame'])
        self.assertGreater(after['timestamp'],before['timestamp'])
        self.assertEqual(before['scene_token'],after['scene_token'])

if __name__=='__main__':unittest.main()
