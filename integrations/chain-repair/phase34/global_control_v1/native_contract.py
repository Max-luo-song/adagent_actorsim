
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
CONTRACT=_hugsim_expand(json.loads((Path(__file__).resolve().parent/"native_contract.json").read_text()))
BOUNDARIES=CONTRACT["comfort_boundaries"]
def calculate_trajectory_kinematics(traj, timestep):
    """
    Calculate kinematic parameters for a given trajectory.

    :param traj: List of tuples (x, y, yaw) for each point in the trajectory
    :param timestep: Time interval between each point in the trajectory
    :return: Dictionary containing lists of calculated parameters
    """
    # Convert trajectory to numpy array for easier calculations
    x, y, yaw = zip(*traj)
    x, y, yaw = np.array(x), np.array(y), np.array(yaw)

    # Calculate velocities
    dx = np.diff(x) / timestep
    dy = np.diff(y) / timestep

    # Calculate yaw rate
    dyaw = np.diff(yaw)
    dyaw = np.where(dyaw > np.pi, dyaw - 2*np.pi, dyaw)
    dyaw = np.where(dyaw < -np.pi, dyaw + 2*np.pi, dyaw)
    dyaw = dyaw / timestep
    ddyaw = np.diff(dyaw) / timestep

    # Calculate speed
    speed = np.sqrt(dx**2 + dy**2)

    # Calculate accelerations
    accel = np.diff(speed) / timestep
    jerk = np.diff(accel) / timestep

    # Calculate yaw rate (already calculated as dyaw)
    yaw_rate = dyaw
    # Calculate yaw acceleration
    yaw_accel = ddyaw

    lon_accel = accel
    lat_accel = np.zeros_like(lon_accel)
    lon_jerk = jerk

    # Pad arrays to match the original trajectory length
    yaw_rate = np.pad(yaw_rate, (0, 1), 'edge')
    yaw_accel = np.pad(yaw_accel, (0, 2), 'edge')
    lon_accel = np.pad(lon_accel, (0, 2), 'edge')
    lat_accel = np.pad(lat_accel, (0, 2), 'edge')
    lon_jerk = np.pad(lon_jerk, (0, 3), 'edge')

    return {
        'speed': speed,
        'yaw_rate': yaw_rate,
        'yaw_accel': yaw_accel,
        'lon_accel': lon_accel,
        'lat_accel': lat_accel,
        'lon_jerk': lon_jerk,
    }

def comfort_terms(traj,dt):
 if len(traj)<4:return dict(proxy=1.,native_flag=1.,max_normalized_ratio=0.)
 k=calculate_trajectory_kinematics(traj,dt);b=BOUNDARIES
 ratios=[np.abs(k['lat_accel'])/b['max_abs_lat_accel'],np.maximum(k['lon_accel'],0)/b['max_lon_accel'],np.maximum(-k['lon_accel'],0)/(-b['min_lon_accel']),np.abs(k['lon_jerk'])/b['max_abs_lon_jerk'],np.abs(k['yaw_accel'])/b['max_abs_yaw_accel'],np.abs(k['yaw_rate'])/b['max_abs_yaw_rate']]
 peak=max(float(x.max()) for x in ratios);native=float(peak<=1.)
 # Continuous optimization pressure starts before the native threshold.
 excess=max(peak-.9,0.)/.1
 return dict(proxy=1./(1.+excess*excess),native_flag=native,max_normalized_ratio=peak)

def combine(nc,dac,ttc,comfort,rc):
 values=np.array([nc,dac,ttc,comfort,rc],dtype=float)
 if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):raise ValueError('Scores must be finite in [0,1]')
 w=CONTRACT['active_weights'];pdms=float(nc*dac*(w['ttc']*ttc+w['c']*comfort)/(w['ttc']+w['c']))
 return dict(pdms=pdms,hdscore=pdms*float(rc))
