"""Offline replay of one closed-loop optimize() call from saved step artifacts.

Read-only on the running arms. Reproduces the exact probe call
  optimize(plan, occupancy, info, warm=None, reliability=guard_memory_report, vehicle=vehicle)
with component timers (TemporalOccupancy build, guard build, per-eval cost,
per-candidate solve) and reports selected action/trajectory/margins for
cross-thread-count consistency checks.
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


def main(step_dir):
    import drivable_area  # preload reviewed phase11 copies (same as run_arm.py)
    import road_route_seed
    import closed172_common as cc
    assert cc.__file__.startswith(str(P11)), cc.__file__
    assert road_route_seed.__file__.startswith(str(P11)), road_route_seed.__file__
    from runtime_configuration import load_vehicle_config

    step = Path(step_dir)
    with np.load(step/'perception.npz') as d:
        plan = d['trajectory'].copy(); occupancy = d['occupancy'].copy()
    with open(step/'sensor_metadata.pkl', 'rb') as f:
        info = pickle.load(f)
    reliability = _hugsim_expand(json.loads((step/'guard_memory_report.json').read_text()))
    vehicle = load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')

    # component timers via monkeypatch (runtime only; files untouched)
    timers = {}
    def wrap(obj, name, key):
        fn = getattr(obj, name)
        def timed(*a, **k):
            t0 = time.perf_counter(); out = fn(*a, **k); dt = time.perf_counter()-t0
            timers.setdefault(key, [0, 0.0]); timers[key][0] += 1; timers[key][1] += dt
            return out
        setattr(obj, name, timed)
    wrap(cc.TemporalOccupancy, '__init__', 'TemporalOccupancy.build')
    wrap(cc.TemporalOccupancy, 'clearance', 'TemporalOccupancy.clearance')
    wrap(cc.PlanningGuard, '__init__', 'PlanningGuard.build')
    wrap(cc.PlanningGuard, 'clearance', 'PlanningGuard.clearance')
    wrap(cc.PlanningGuard, 'soft_cost', 'PlanningGuard.soft_cost')
    wrap(cc, 'rollout', 'rollout')
    from scipy import optimize as _so
    _min = _so.minimize
    def timed_min(*a, **k):
        t0 = time.perf_counter(); out = _min(*a, **k); dt = time.perf_counter()-t0
        timers.setdefault('SLSQP.solve', [0, 0.0]); timers['SLSQP.solve'][0] += 1; timers['SLSQP.solve'][1] += dt
        return out
    _so.minimize = timed_min
    import drivable_area as _da
    wrap(_da.DrivableArea, '__init__', 'DrivableArea.build')
    wrap(_da.DrivableArea, 'clearance', 'DrivableArea.clearance')
    wrap(_da.DrivableArea, 'swept_clearance', 'DrivableArea.swept_clearance')

    t0 = time.perf_counter()
    result = cc.optimize(plan, occupancy, info, warm=None, reliability=reliability, vehicle=vehicle)
    total = time.perf_counter()-t0

    trials = result.get('trials', [])
    sel = result.get('selected')
    summary = dict(
        step=str(step), total_s=round(total, 2), candidates=len(trials),
        feasible=sum(1 for t in trials if t.get('feasible')),
        reason=result.get('reason'),
        selected_seed=(sel or {}).get('seed'),
        selected_controls=(sel or {}).get('controls'),
        selected_clearance=(sel or {}).get('clearance_lower_bound_m'),
        selected_min_constraint=(sel or {}).get('minimum_constraint'),
        timers=timers)
    return summary


if __name__ == '__main__':
    s = main(sys.argv[1])
    print(json.dumps(s, indent=1))
