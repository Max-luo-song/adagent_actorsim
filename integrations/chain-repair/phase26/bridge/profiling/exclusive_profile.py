"""Exclusive (self-time) profiling of the clearance chain via call-stack accounting.

Each wrapped function's self time = its wall time minus time spent in wrapped
children. Answers whether road/occupancy/guard clearance were double-counted
through nesting in the earlier inclusive profile.
"""

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

import json
import pickle
import sys
import time
pass  # Path is provided by the portable boundary above

import numpy as np

JOB = Path(_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4'))
P11 = JOB/'chain_repair/phase11'
ROOT = JOB.parents[3]
sys.path[:0] = [str(P11), str(ROOT/'integrations/sr-hugsim'), str(ROOT/'waymo23-native-import'),
                str(ROOT/'waymo23-native-import/waymo172_closed_loop_v1')]

import drivable_area
import road_route_seed
import closed172_common as cc
from runtime_configuration import load_vehicle_config

STACK_CHILD = []
SELF = {}
TOT = {}


def wrap2(obj, name, key):
    fn = getattr(obj, name)
    def timed(*a, **k):
        t0 = time.perf_counter()
        STACK_CHILD.append(0.0)
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter()-t0
            child = STACK_CHILD.pop()
            SELF.setdefault(key, [0, 0.0]); SELF[key][0] += 1; SELF[key][1] += dt-child
            TOT.setdefault(key, [0, 0.0]); TOT[key][0] += 1; TOT[key][1] += dt
            if STACK_CHILD:
                STACK_CHILD[-1] += dt
    setattr(obj, name, timed)


def main(step_dir):
    step = Path(step_dir)
    with np.load(step/'perception.npz') as d:
        plan = d['trajectory'].copy(); occupancy = d['occupancy'].copy()
    with open(step/'sensor_metadata.pkl', 'rb') as f:
        info = pickle.load(f)
    reliability = _hugsim_expand(json.loads((step/'guard_memory_report.json').read_text()))
    vehicle = load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')

    wrap2(cc.TemporalOccupancy, 'clearance', 'TemporalOccupancy.clearance')
    wrap2(cc.PlanningGuard, 'clearance', 'PlanningGuard.clearance')
    wrap2(cc.PlanningGuard, 'soft_cost', 'PlanningGuard.soft_cost')
    wrap2(cc, 'rollout', 'rollout')
    import planning_guard as pg
    wrap2(pg, 'occupancy_clearance', 'occupancy_clearance')
    wrap2(pg, 'rectangle_separation', 'rectangle_separation')
    _da = drivable_area.DrivableArea
    wrap2(_da, 'clearance', 'DrivableArea.clearance')
    wrap2(_da, 'swept_clearance', 'DrivableArea.swept_clearance')

    t0 = time.perf_counter()
    result = cc.optimize(plan, occupancy, info, warm=None, reliability=reliability, vehicle=vehicle)
    total = time.perf_counter()-t0
    out = dict(step=str(step), total_s=round(total, 2),
               exclusive={k: dict(calls=v[0], self_s=round(v[1], 2)) for k, v in sorted(SELF.items(), key=lambda kv: -kv[1][1])},
               inclusive={k: dict(calls=v[0], total_s=round(v[1], 2)) for k, v in sorted(TOT.items(), key=lambda kv: -kv[1][1])})
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
