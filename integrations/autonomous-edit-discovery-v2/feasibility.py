"""Offline-only witness checks on recorded paths; never supplied to the controller."""

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
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from shapely.geometry import Polygon
from common_flow import read,asset

def rectangle(pose,length,width):
 q=np.array([[-length/2,-width/2,0],[length/2,-width/2,0],[length/2,width/2,0],[-length/2,width/2,0]])
 return Polygon((q@pose[:3,:3].T+pose[:3,3])[:,:2])

def check(taskfile,witness_report):
 witness_report=Path(witness_report);task=read(taskfile);r=read(witness_report)
 if r['status']!='completed' or len(r['trace'])!=188:raise ValueError('Witness must be an actual complete prior path')
 if any(o['environment']['collision'] or o['environment'].get('authorized_static_map_departure') for o in r['observations']):raise ValueError('Witness path was not clear in the unchanged static world')
 with np.load(task['edit_assets']+'/target_track.npz') as z:
  trans=z['rendered_trans'];quats=z['rendered_quats']
 with np.load(str(witness_report.parent/'actor_tracks.npz')) as z:size=z['RigidNodes__instances_size'][3]
 poses=np.asarray([o['pose'] for o in r['observations']]);rots=Rotation.from_matrix(poses[:,:3,:3]);rotation=Slerp(np.arange(len(poses))*.1,rots)
 vehicle=r['vehicle_geometry'];minimum=float('inf');close_at=None
 for timestamp in np.arange(0,18.80001,.01):
  timestamp=min(float(timestamp),18.8);x=timestamp*10;lo=min(int(x),187);u=x-lo;pose=np.eye(4);pose[:3,:3]=rotation([timestamp])[0].as_matrix();pose[:3,3]=poses[lo,:3,3]*(1-u)+poses[lo+1,:3,3]*u
  frame=10+x;i=min(int(frame),197);v=frame-i;actor=np.eye(4);actor[:3,3]=trans[i]*(1-v)+trans[i+1]*v
  ar=Slerp([0.,1.],Rotation.from_quat(quats[[i,i+1]][:,[1,2,3,0]]))([v])[0];actor[:3,:3]=ar.as_matrix()
  ego=rectangle(pose,2*vehicle['half_length_m'],2*vehicle['half_width_m']);other=rectangle(actor,size[0],size[1]);gap=ego.distance(other)
  if gap<minimum:minimum=float(gap);close_at=float(timestamp)
 # The edit generator verifies other actors and unchanged source identity. This
 # certifies an engineering witness only, not independent physical scene truth.
 passed=minimum>=.15
 return dict(passed=passed,minimum_edited_actor_gap_m=minimum,closest_local_s=close_at,source_path=asset(witness_report),edit=task['edit_receipts'],scope='offline replayed complete path checked against the NEW edited actor at 10ms; same static world and other actors',given_to_online_controller=False,physical_truth_certified=False)
