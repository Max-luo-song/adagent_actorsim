
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

pass  # Path is provided by the portable boundary above
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import tracking_common as tc
import planning_guard_tracking as pgt
from driver_state_audit import initial_driver_state
from depth_sensor_map import DepthObstacleMemory
from road_prediction_memory import RoadPredictionMemory
from navigation_anchor import NavigationAnchor
from committed_route import CommittedRoute


class DriverTests(unittest.TestCase):
    def setUp(self):
        vehicle=tc.load_vehicle_config()
        self.motion=SimpleNamespace(x=0.,z=0.,yaw=0.,last_accel=0.,last_rate=0.,speed=3.,steer=.01,timestamp=8.)
        self.controller=pgt.RecoveryController();self.obstacles=pgt.ObstacleMemory()
        self.depth=DepthObstacleMemory(vehicle);self.road=RoadPredictionMemory(vehicle)
        self.nav=NavigationAnchor();self.route=CommittedRoute()
    def audit(self,warm=None):
        return initial_driver_state(self.motion,warm,self.controller,self.obstacles,self.depth,self.road,self.nav,self.route)
    def test_fresh_actual_control_objects_have_empty_history(self):
        result=self.audit()
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['initial_speed_mps'],3.)  # source initial speed is preserved
    def test_each_planning_memory_leak_is_visible(self):
        self.controller.previous_time=9.;self.obstacles.tracks.append({})
        self.depth.cells[1]={};self.depth.visibility.hypotheses.append({})
        self.road.history.append({});self.nav.points=[1];self.route.world=[1]
        result=self.audit(warm=[1])['checks']
        for key in ('warm_empty','recovery_fresh','obstacle_history_empty','depth_history_empty',
                    'visibility_history_empty','road_history_empty','navigation_empty','committed_route_empty'):
            self.assertFalse(result[key],key)


if __name__=='__main__':unittest.main()
