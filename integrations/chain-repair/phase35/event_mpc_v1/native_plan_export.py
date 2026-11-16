"""Export the adopted control plan for native evaluation; no scene truth input."""

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
from empty_controller import integrate_state

def save_plan(folder,selected,following,info,vehicle,fallback=None):
 c=np.asarray(info['model_lidar_frame']['virtual_to_ego']);r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t'])
 if following and selected is not None:
  states=np.asarray(selected['states']);kind='adopted_mpc'
 elif fallback is not None:
  states=np.asarray(fallback['states']);kind='joint_braking_steering_fallback'
 else:
  states=[np.r_[-c[:2,3],0.,info['ego_velo'],info['ego_steer']]]
  for _ in range(int(round(6./vehicle.dt))):
   previous=states[-1];acc=-min(vehicle.emergency_braking_mps2,previous[3]/vehicle.dt);states.append(integrate_state(previous,[acc,0.],vehicle.dt,vehicle.wheelbase_m))
  states=np.asarray(states);kind='controlled_braking_fallback'
 nodes=states[::int(round(.5/vehicle.dt))]
 world=np.c_[nodes[:,:2],np.full(len(nodes),-c[2,3])]@r.T+t
 xy=world[:,[1,0]]*np.array([1.,-1.]);forward=np.c_[np.sin(nodes[:,2]),np.cos(nodes[:,2]),np.zeros(len(nodes))]@r.T;yaw=np.arctan2(-forward[:,0],forward[:,1]);traj=np.c_[xy,yaw]
 payload=dict(schema='native_adopted_plan_v1',time_stamp=float(info['timestamp']),is_key_frame=True,ego_box=[float(xy[0,0]),float(xy[0,1]),float(world[0,2]),2*vehicle.half_width_m,2*vehicle.half_length_m,vehicle.body_height_m,float(yaw[0])],planned_traj=dict(traj=traj[1:].tolist(),timestep=.5),plan_kind=kind,frame='hugsim_score_imu_world',scene_truth_used_for_control=False)
 path=Path(folder)/'native_adopted_plan.json';path.write_text(json.dumps(payload,indent=2)+chr(10));return payload
