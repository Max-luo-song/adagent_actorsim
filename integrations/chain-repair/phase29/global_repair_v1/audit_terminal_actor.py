"""Post-hoc actor contact audit; reconstruction truth never enters planning."""

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
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(CR/'phase28'),str(ROOT/'integrations/sr-hugsim'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from native_actor_evaluation import NativeActorEvaluation
from waymo_adapter import SR_TO_HUGSIM as T
from heading_fix_v1.closed172_common import load_vehicle_config
S=np.array([[0.,-1,0],[1,0,0],[0,0,1]])

def main():
 run=CR/'phase28/runs/lineage92_closed_scene552_v4';report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));vehicle=load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
 actors=NativeActorEvaluation.from_archive(dict(np.load(CR/'phase5/native552_actor_geometry.npz')))
 pose=np.array(report['observations'][-1]['closed_ego_to_world']);contacts=[]
 for fraction in np.linspace(0,1,11):
  check=actors.check(pose,184+fraction,vehicle)
  contacts.append(dict(source_frame=184+float(fraction),**check))
 target=next(row for row in contacts if row['overlap'])['overlaps'][0]
 group=target['group'];identity=target['object_index'];rows=[]
 for index in range(85,105):
  d=run/f'uniad_worker/step_{index:03d}'
  if not (d/'perception.npz').exists():continue
  frame=80+index;actor=next(r for r in actors.geometry(float(frame)) if r[0]==group and r[1]==identity)
  poly=actor[2];center=np.r_[poly.mean(0),(actor[3]+actor[4])/2]
  prev=next(r for r in actors.geometry(float(frame)-1) if r[0]==group and r[1]==identity)
  velocity=(center[:2]-prev[2].mean(0))/.1
  with (d/'observation.pkl').open('rb') as f:_,info=pickle.load(f)
  r=np.array(info['explicit_l2g_r_mat']);t=np.array(info['explicit_l2g_t']);local=(center@S.T-t)@r
  reliability=_hugsim_expand(json.loads((d/'guard_memory_report.json').read_text()));origin=_hugsim_expand(json.loads((d/'neural_query_origin.json').read_text()))
  with np.load(d/'perception.npz') as arrays:
   gravity=arrays['candidate_boxes_gravity'];dist=np.linalg.norm(gravity[:,:2]-local[:2],axis=1);order=np.argsort(dist)[:3]
   matched=[]
   for i in order:
    obs=[o for o in reliability['obstacles'] if o['candidate_index']==int(i)]
    matched.append(dict(index=int(i),neural_id=origin['selected_track_ids'][i],label=int(arrays['candidate_labels'][i]),score=float(arrays['candidate_scores'][i]),distance_xy_m=float(dist[i]),box=arrays['candidate_boxes'][i].tolist(),retained_report=obs))
  rows.append(dict(step=index,source_frame=frame,actor_center_source=center.tolist(),actor_velocity_source_xy_mps=velocity.tolist(),actor_local_center=local.tolist(),ego_speed=info['ego_velo'],nearest_candidates=matched))
 out=CR/'phase29/runs/terminal_actor_audit';out.mkdir(exist_ok=True)
 (out/'audit.json').write_text(json.dumps(dict(scope='Post-hoc reconstruction actor proxy audit, not mesh contact certification or control input',target=target,interpolated_contacts=contacts,frames=rows),indent=2)+'\n')
 print('TARGET',target)
 for row in rows[::3]:print(row['step'],'actor speed',np.linalg.norm(row['actor_velocity_source_xy_mps']),'nearest',[(m['index'],m['label'],m['score'],m['distance_xy_m'],bool(m['retained_report'])) for m in row['nearest_candidates']])
if __name__=='__main__':main()
