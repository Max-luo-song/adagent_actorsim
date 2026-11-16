from bootstrap import *
from run_driver import configure
import empty_controller,depth_sensor_map
from depth_consistency import DepthConsistencyMemory
from speed_policy import PlanningGuard
configure()
assert base.boundary_optimize is empty_controller.optimize
assert empty_controller.PlanningGuard is PlanningGuard
assert depth_sensor_map.DepthObstacleMemory is DepthConsistencyMemory
assert h.WORKER==HERE/'worker.py'
assert h.optimize.__name__=='audited_optimize'
print('New empty occupancy controller, evidence guard, depth policy and audited live driver connected; no GPU loaded')
