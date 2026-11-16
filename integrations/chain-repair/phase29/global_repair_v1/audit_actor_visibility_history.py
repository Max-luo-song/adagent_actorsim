
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
import json,pickle,sys
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from input_geometry import intrinsic_matrix
S=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
run=CR/'phase28/runs/lineage92_closed_scene552_v4'
with np.load(CR/'phase5/native552_actor_geometry.npz') as a:
 trans=a['RigidNodes__instances_trans'];quat=a['RigidNodes__instances_quats'];size=a['RigidNodes__instances_size'][7]
rows=[]
for i in range(105):
 d=run/f'uniad_worker/step_{i:03d}';f=d/'sensor_metadata.pkl'
 if f.exists():
  with f.open('rb') as stream:info=pickle.load(stream)
 else:
  with (d/'observation.pkl').open('rb') as stream:_,info=pickle.load(stream)
 frame=80+i;q=quat[frame,7].reshape(4);rotation=Rotation.from_quat(q[[1,2,3,0]])
 points=np.array([[x*size[0]/2,y*size[1]/2,z*size[2]/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
 points=np.vstack([np.zeros(3),points]);world=rotation.apply(points)+trans[frame,7]
 local=(world@S.T-np.array(info['explicit_l2g_t']))@np.array(info['explicit_l2g_r_mat'])
 cameras={}
 for name,c in info['cam_params'].items():
  cp=(np.array(c['l2c'])@np.c_[local,np.ones(len(local))].T).T[:,:3];p=cp@intrinsic_matrix(c['intrinsic'])[:3,:3].T
  uv=p[:,:2]/np.maximum(cp[:,2,None],1e-8);inside=(cp[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<c['intrinsic']['W'])&(uv[:,1]>=0)&(uv[:,1]<c['intrinsic']['H'])
  cameras[name]=np.flatnonzero(inside).tolist()
 rows.append(dict(step=i,source_frame=frame,source_center=trans[frame,7].tolist(),cameras=cameras,any_center_visible=any(0 in v for v in cameras.values()),any_corner_visible=any(bool(v) for v in cameras.values())))
out=CR/'phase29/runs/terminal_actor_audit/actor_visibility_history.json';out.write_text(json.dumps(rows,indent=2)+'\n')
print('Actor center visible steps:',[r['step'] for r in rows if r['any_center_visible']])
print('Any actor point visible steps:',[r['step'] for r in rows if r['any_corner_visible']])
