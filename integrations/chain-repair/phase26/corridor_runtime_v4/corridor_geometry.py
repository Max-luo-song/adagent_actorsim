"""Forward bypass constraints and explicit side identity, not obstacle removal."""

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
pass  # Path is provided by the portable boundary above
import numpy as np
from native_contract import CONTRACT
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name('corridor_policy.json').read_text()))
def forward_constraints(states,vehicle):
 states=np.asarray(states);bounds=CONTRACT['comfort_boundaries'];lat=np.abs(states[:,3]**2*np.tan(states[:,4])/vehicle.wheelbase_m);yawrate=np.abs(states[:,3]*np.tan(states[:,4])/vehicle.wheelbase_m)
 # Do not demand an instantaneous correction of an already excessive initial
 # steering state. Fresh safe initial states retain the native comfort limits.
 lat_limit=max(bounds['max_abs_lat_accel'],float(lat[0]));yaw_limit=max(bounds['max_abs_yaw_rate'],float(yawrate[0]))
 return np.r_[POLICY['maximum_heading_rad']-np.abs(states[:,2]),np.diff(states[:,1]),lat_limit-lat[1:],yaw_limit-yawrate[1:]]
def side_constraints(states,initial,spec):
 if spec is None:return np.empty(0)
 states=np.asarray(states);sign=spec['sign'];tol=POLICY['side_tolerance_m'];displacement=sign*(states[:,0]-initial[0]);anchor=spec['anchor_index']
 required=max(0.,abs(spec['displacement_m'])-tol)
 # Keep the chosen side after the anchor until the whole body has advanced
 # beyond the current blocking geometry. A constant-sized OR constraint avoids
 # changing SLSQP's constraint dimensions as the passing time changes.
 hold=np.maximum(states[anchor:,1]-spec['release_forward_m'],displacement[anchor:]-required)
 return np.r_[displacement[1:]+tol,displacement[anchor]-required,hold]
