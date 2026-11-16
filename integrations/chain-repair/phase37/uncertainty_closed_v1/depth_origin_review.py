
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
import json,pickle,numpy as np
from depth_surface_geometry import ground_plane,points_from_depth
source=CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop/uniad_worker'
rows=[]
for folder in sorted(source.glob('step_*')):
 if not all((folder/name).is_file() for name in ('sensor_metadata.pkl','guard_memory_report.json','surface_depth.npz','surface_valid.npz','observed_depth.npz')):continue
 with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
 obs={}
 for name,filename in [('depth','observed_depth.npz'),('depth_variance','observed_depth_variance.npz'),('depth_alpha','observed_depth_alpha.npz'),('surface_depth','surface_depth.npz'),('surface_valid','surface_valid.npz')]:
  if (folder/filename).exists():
   with np.load(folder/filename) as z:obs[name]={k:z[k].copy() for k in z.files}
 plane=ground_plane(obs,info);report=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));points=points_from_depth(obs,info,6)
 r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t']);cells=report.get('observed_depth_geometry',{}).get('cell_evidence',[]);corridor=[]
 # Diagnostic corridor derived from actual ego dimensions and current speed.
 # No selected coordinate or target ID enters the online planner.
 for cell in cells:
  if not cell.get('hard_geometry'):continue
  center=((np.asarray(cell['world_voxel'])+.5)*.5-t)@r
  if abs(center[0])>1.3 or not 0<center[1]<max(1.,float(info['ego_velo'])*6.):continue
  close=points[np.linalg.norm(points[:,:2]-center[:2],axis=1)<=2.]
  corridor.append(dict(world_voxel=cell['world_voxel'],local_center=center.tolist(),height_evidence=cell.get('height_evidence'),neighbor_depth_samples=len(close),neighbor_height_range=(np.quantile(close[:,2],[0.,.5,1.]).tolist() if len(close) else None),plane_ground_height_m=(float(np.r_[center[:2],1.]@plane) if plane is not None else None)))
 rows.append(dict(step=folder.name,timestamp=float(info['timestamp']),ground_plane=(plane.tolist() if plane is not None else None),strict_surface_available=True,depth_samples=len(points),hard_corridor_cells=corridor))
summary=dict(frames=rows,scope='Sensor-only diagnostic of depth/ground reference. A height-varying surface is not certified empty road.',online_geometry_modified=False,root_cause_resolved=False,new_gpu_executed=False,remaining='A global plane and fixed height-error budget do not establish a local road-height bound. Do not erase elevated cells without independent support or mark this review accepted.')
(HERE/'depth_origin_review.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(frames=len(rows),frames_with_hard_corridor_cells=sum(bool(row['hard_corridor_cells']) for row in rows),root_cause_resolved=False,online_geometry_modified=False),indent=2))
