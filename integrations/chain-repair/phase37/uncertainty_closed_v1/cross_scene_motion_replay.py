
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
import json,pickle,copy
import numpy as np
from motion_memory import ContractObstacleMemory
from speed_policy import PlanningGuard
from evidence_guard import PlanningGuard as PreviousGuard
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
v=load_vehicle_config();rows=[]
for scene,source in [(551,CR/'phase34/runs/global_control551_v2_20261106T141946Z_4f9bf305/closed_loop'),(552,CR/'phase29/runs/global_evidence_closed552_v2')]:
 spec=h.SceneSpec.read(source/'scene_manifest.json');memory=ContractObstacleMemory.from_source(spec.source);frames=[];incomplete=[]
 for folder in sorted((source/'uniad_worker').glob('step_*')):
  if not all((folder/name).is_file() for name in ('scene_reliability.json','sensor_metadata.pkl','perception.npz')):
   incomplete.append(folder.name);continue
  with (folder/'sensor_metadata.pkl').open('rb') as stream:info=pickle.load(stream)
  with np.load(folder/'perception.npz') as saved:
   if 'occupancy_contract_json' in saved:
    occ,_=unpack(saved)
   else:
    # Offline-only legacy reader: never fabricate a worker availability receipt.
    occ=saved['occupancy'].copy()
    if occ.size!=5*200*200 or not np.isfinite(occ).all():raise ValueError('Invalid archived legacy occupancy')
   boxes=saved['geometry_candidate_boxes'].copy() if 'geometry_candidate_boxes' in saved else saved['candidate_boxes'].copy();scores=saved['candidate_scores'].copy();labels=saved['candidate_labels'].copy()
  original=_hugsim_expand(json.loads((folder/'scene_reliability.json').read_text()));before=copy.deepcopy(original);arrays=[x.copy() for x in (boxes,scores,labels)];report=memory.update(original,boxes,scores,labels,info,current_occupancy=np.asarray(occ).reshape(5,200,200)[0] if occ is not None else None)
  assert original==before
  for a,b in zip(arrays,(boxes,scores,labels)):np.testing.assert_array_equal(a,b)
  times=np.arange(31)*v.dt;states=np.zeros((31,5));states[:,1]=times*float(info['ego_velo']);states[:,3]=info['ego_velo'];current=np.asarray(occ).reshape(5,200,200)[0] if occ is not None else None
  old=PreviousGuard(report,v,states,times,current_occupancy=current);new=PlanningGuard(report,v,states,times,current_occupancy=current)
  np.testing.assert_array_equal(old.hard_boxes,new.hard_boxes);np.testing.assert_array_equal(old.hard_margins,new.hard_margins)
  if not new.local_motion:np.testing.assert_array_equal(old.clearance(states,times),new.clearance(states,times))
  frames.append(dict(frame=folder.name,qualified_targets=len(new.local_motion),hard_targets=len(new.hard_boxes),stationary_or_unqualified_fallback_exact=not new.local_motion))
 rows.append(dict(scene=scene,frames=len(frames),incomplete_frames=incomplete,qualified_frame_count=sum(x['qualified_targets']>0 for x in frames),qualified_target_observations=sum(x['qualified_targets'] for x in frames),details=frames))
summary=dict(scenes=rows,raw_arrays_unchanged=True,current_hard_geometry_and_margins_unchanged=True,scope='Archived-observation motion/memory replay; no new closed loop, no cross-scene avoidance success claim.',new_gpu_executed=False)
(HERE/'cross_scene_motion_replay.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({**summary,'scenes':[{k:v for k,v in row.items() if k!='details'} for row in rows]},indent=2))
