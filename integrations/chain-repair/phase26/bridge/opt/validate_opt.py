"""Validate the optimized clearance/optimize copy against the original.

Three saved steps: normal feasible (20), near-boundary (25), all-infeasible
slow (26). Compares: raw clearance outputs (bit level), per-trial constraints,
solve result, selected action/trajectory, minimum margins, plus wall time and
peak memory. Any feasibility flip is reported as an error, not waved through.
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
import tracemalloc
pass  # Path is provided by the portable boundary above

import numpy as np

JOB = Path(_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4'))
P11 = JOB/'chain_repair/phase11'
OPT = Path(__file__).resolve().parent
ROOT = JOB.parents[3]
sys.path[:0] = [str(OPT), str(P11), str(ROOT/'integrations/sr-hugsim'),
                str(ROOT/'waymo23-native-import'), str(ROOT/'waymo23-native-import/waymo172_closed_loop_v1')]

import drivable_area  # phase11 reviewed copies first (mirrors run_arm.py)
import road_route_seed
import closed172_common as cc_orig
import closed172_common_opt as cc_opt
from runtime_configuration import load_vehicle_config

assert cc_orig.__file__.startswith(str(P11))
assert cc_opt.__file__.startswith(str(OPT))
import temporal_occupancy_fast
assert temporal_occupancy_fast.__file__.startswith(str(OPT))


def load_step(step_dir):
    step = Path(step_dir)
    with np.load(step/'perception.npz') as d:
        plan = d['trajectory'].copy(); occupancy = d['occupancy'].copy()
    with open(step/'sensor_metadata.pkl', 'rb') as f:
        info = pickle.load(f)
    reliability = _hugsim_expand(json.loads((step/'guard_memory_report.json').read_text()))
    return plan, occupancy, info, reliability


def clearance_level_check(plan, occupancy, info, reliability, vehicle):
    """Bit-level comparison of clearance outputs on shared random state batches."""
    masks = np.asarray(occupancy).squeeze(2)
    occ_args = dict(
        observed_cell_rotation=info.get('explicit_l2g_r_mat') if (reliability.get('observed_depth_geometry', {}).get('policy') == 'observed_depth_geometry_with_free_space_clearing_v1' and np.isclose(reliability.get('observed_depth_geometry', {}).get('voxel_resolution_m', -1), .5)) else None,
        observed_points=reliability.get('observed_depth_obstacles_xy'),
        observed_point_radius=reliability.get('observed_depth_geometry', {}).get('voxel_half_diagonal_m', np.sqrt(2)*.25),
        unknown_cell_rotation=info.get('explicit_l2g_r_mat') if (reliability.get('unknown_visibility_geometry', {}).get('policy') == 'endpoint_supported_unknown_visibility_v2' and np.isclose(reliability.get('observed_depth_geometry', {}).get('voxel_resolution_m', -1), .5)) else None,
        unknown_points=reliability.get('unknown_visibility_obstacles_xy'),
        unknown_point_radius=reliability.get('unknown_visibility_geometry', {}).get('point_radius_m', np.sqrt(2)*.25))
    orig = cc_orig.TemporalOccupancy(masks, vehicle, **occ_args)
    fast = cc_opt.TemporalOccupancy(masks, vehicle, **occ_args)
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(6):
        states = rng.normal(size=(40, 5))*[8., 8., .3, 3., .1]
        times = np.sort(rng.random(40))*3.
        a = orig.clearance(states, times)
        b = fast.clearance(states, times)
        err = float(np.abs(a-b).max())
        worst = max(worst, err)
        assert a.shape == b.shape
    return worst


def run_optimize(cc, plan, occupancy, info, reliability, vehicle):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = cc.optimize(plan, occupancy, info, warm=None, reliability=reliability, vehicle=vehicle)
    total = time.perf_counter()-t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, total, peak


def compare_results(a, b):
    out = {}
    ta, tb = a.get('trials', []), b.get('trials', [])
    out['trial_count'] = (len(ta), len(tb))
    flips = []
    max_dc = max_dcons = 0.0
    for ra, rb in zip(ta, tb):
        if ra['seed'] != rb['seed']:
            raise ValueError('trial order changed')
        if bool(ra['feasible']) != bool(rb['feasible']):
            flips.append(ra['seed'])
        max_dc = max(max_dc, float(np.abs(np.asarray(ra['controls'])-np.asarray(rb['controls'])).max()))
        max_dcons = max(max_dcons, abs(ra['minimum_constraint']-rb['minimum_constraint']))
    sel_a, sel_b = a.get('selected'), b.get('selected')
    out['feasibility_flips'] = flips
    out['max_controls_diff'] = max_dc
    out['max_min_constraint_diff'] = max_dcons
    out['selected_seed'] = ((sel_a or {}).get('seed'), (sel_b or {}).get('seed'))
    if sel_a and sel_b:
        out['selected_controls_max_diff'] = float(np.abs(np.asarray(sel_a['controls'])-np.asarray(sel_b['controls'])).max())
        out['selected_clearance_diff'] = abs(sel_a['clearance_lower_bound_m']-sel_b['clearance_lower_bound_m'])
        sa = np.asarray(sel_a['states']); sb = np.asarray(sel_b['states'])
        out['selected_trajectory_max_diff_m'] = float(np.abs(sa[:, :2]-sb[:, :2]).max())
    out['reason'] = (a.get('reason'), b.get('reason'))
    return out


if __name__ == '__main__':
    steps = sys.argv[1:]
    vehicle = None
    # fix: keep references tidy
    results = {}
    vehicle = load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    for s in steps:
        plan, occupancy, info, reliability = load_step(s)
        clear_err = clearance_level_check(plan, occupancy, info, reliability, vehicle)
        ra, ta, ma = run_optimize(cc_orig, plan, occupancy, info, reliability, vehicle)
        rb, tb, mb = run_optimize(cc_opt, plan, occupancy, info, reliability, vehicle)
        cmp_ = compare_results(ra, rb)
        entry = dict(clearance_bit_max_diff=clear_err,
                     original_s=round(ta, 2), optimized_s=round(tb, 2),
                     original_peak_mb=round(ma/1e6, 1), optimized_peak_mb=round(mb/1e6, 1),
                     speedup=round(ta/tb, 2), **cmp_)
        results[Path(s).parent.name+'/'+Path(s).name] = entry
        print(Path(s).name, json.dumps(entry, default=str)[:300], flush=True)
        # incremental write: survive interruption; never fabricate missing steps
        (Path(__file__).parent/'opt_validation.json').write_text(json.dumps(results, indent=1, default=str))
    out = Path(__file__).parent/'opt_validation.json'
    print('written', out)
