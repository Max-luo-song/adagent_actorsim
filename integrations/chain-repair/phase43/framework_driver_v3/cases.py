
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
from runtime_configuration import load_vehicle_config
import json,pickle,numpy as np
SOURCE=CR/'phase31/runs/stopped_vehicle551_20261105T030420Z_e8495703/closed_loop'

def load(index):
 folder=SOURCE/f'uniad_worker/step_{index:03d}'
 with (folder/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
 rel=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
 with np.load(folder/'perception.npz') as f:plan=f['trajectory'].copy();occ=f['occupancy'].copy()
 vehicle=load_vehicle_config();offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
 states=[np.array([-offset[0],-offset[1],0.,info['ego_velo'],info['ego_steer']])]
 for i in range(60):
  acc=0. if i*vehicle.dt<rel['config']['confirmation_seconds']-1e-9 else -min(vehicle.emergency_braking_mps2,states[-1][3]/vehicle.dt)
  states.append(h.integrate_state(states[-1],[acc,0.],vehicle.dt,vehicle.wheelbase_m))
 return folder,rel,info,plan,occ,vehicle,np.asarray(states),np.arange(len(states))*vehicle.dt
