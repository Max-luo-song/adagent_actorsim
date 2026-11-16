"""Step-026 divergence diagnosis: compare orig vs opt on identical fixed inputs.

1. Clearance on dense state batches near the inserted car (bit level).
2. constraints/objective at every SLSQP seed start (bit level).
3. One full SLSQP solve for the first diverging-relevant seed with eval tracing:
   first input point where constraints outputs differ.
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

import json, pickle, sys
pass  # Path is provided by the portable boundary above
import numpy as np

JOB = Path(_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4'))
P11 = JOB/'chain_repair/phase11'
OPT = Path(_hugsim_expand('${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/opt'))
ROOT = JOB.parents[3]
sys.path[:0] = [str(OPT), str(P11), str(ROOT/'integrations/sr-hugsim'), str(ROOT/'waymo23-native-import')]
from runtime_configuration import load_vehicle_config

step = JOB/'chain_repair/phase26/bridge/closed_arms/arm_A/uniad_worker/step_026'
with np.load(step/'perception.npz') as d:
    occupancy = d['occupancy'].copy()
with open(step/'sensor_metadata.pkl','rb') as f:
    info = pickle.load(f)
rel = _hugsim_expand(json.loads((step/'guard_memory_report.json').read_text()))
vehicle = load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')

import closed172_common as cco
import closed172_common_opt as ccf
masks = np.asarray(occupancy).squeeze(2)
kwargs = dict(
    observed_cell_rotation=info.get('explicit_l2g_r_mat') if (rel.get('observed_depth_geometry',{}).get('policy')=='observed_depth_geometry_with_free_space_clearing_v1' and np.isclose(rel.get('observed_depth_geometry',{}).get('voxel_resolution_m',-1),.5)) else None,
    observed_points=rel.get('observed_depth_obstacles_xy'),
    observed_point_radius=rel.get('observed_depth_geometry',{}).get('voxel_half_diagonal_m',np.sqrt(2)*.25),
    unknown_cell_rotation=info.get('explicit_l2g_r_mat') if (rel.get('unknown_visibility_geometry',{}).get('policy')=='endpoint_supported_unknown_visibility_v2' and np.isclose(rel.get('observed_depth_geometry',{}).get('voxel_resolution_m',-1),.5)) else None,
    unknown_points=rel.get('unknown_visibility_obstacles_xy'),
    unknown_point_radius=rel.get('unknown_visibility_geometry',{}).get('point_radius_m',np.sqrt(2)*.25))
o = cco.TemporalOccupancy(masks, vehicle, **kwargs)
f = ccf.TemporalOccupancy(masks, vehicle, **kwargs)

# 1) dense batches near the inserted car (world anchor ~ (23.3, 3.5); model frame: ego-local)
rng = np.random.default_rng(7)
worst = 0.0; worst_where=None
for trial in range(40):
    n = 60
    states = np.zeros((n,5))
    states[:,0] = rng.uniform(-6, 6, n)      # right
    states[:,1] = rng.uniform(5, 30, n)      # forward
    states[:,2] = rng.uniform(-np.pi, np.pi, n)
    states[:,3] = rng.uniform(0, 4, n)
    times = np.sort(rng.random(n))*3.
    a = o.clearance(states, times); b = f.clearance(states, times)
    err = float(np.abs(a-b).max())
    if err > worst: worst, worst_where = err, trial
print('dense-near-obstacle clearance max bit diff:', worst, 'at trial', worst_where)

# 2) also wide-range batches (like solve excursions)
for trial in range(20):
    n=80
    states = np.zeros((n,5))
    states[:,0] = rng.uniform(-50, 50, n); states[:,1] = rng.uniform(-50, 50, n)
    states[:,2] = rng.uniform(-np.pi, np.pi, n)
    times = np.sort(rng.random(n))*3.
    a = o.clearance(states, times); b = f.clearance(states, times)
    err = float(np.abs(a-b).max())
    if err: print('wide batch', trial, 'diff', err)
print('wide-range batches done (no print = bit-identical)')
