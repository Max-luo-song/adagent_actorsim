"""Saved-input inference screening; detector-reference matching is not truth."""

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

import json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(P28),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from heading_fix_v1.trace_near_ego import guard
from heading_fix_v1 import closed172_common as cc
from input_geometry import intrinsic_matrix

def center_visible(box,info):
    point=np.r_[box[:3]+[0,0,box[5]/2],1.]
    for camera in info['cam_params'].values():
        cp=np.asarray(camera['l2c'])@point;k=intrinsic_matrix(camera['intrinsic']);uv=k[:3,:3]@cp[:3]
        if cp[2]>.1 and 0<=uv[0]/cp[2]<camera['intrinsic']['W'] and 0<=uv[1]/cp[2]<camera['intrinsic']['H']:return True
    return False

def assess(output):
    output=Path(output);vehicle=cc.load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    result=dict(formal_evaluation_admitted=False,closed_loop_success_verified=False,reference='Baseline detector center-visible retained objects; not physical ground truth',cases={},visible_retained_reference_objects=0,visible_retained_reference_misses=[])
    for mode in ('baseline','repair'):
        conflicts=[]
        for index in range(81):
            d=output/mode/f'step_{index:03d}'
            with (d/'observation.pkl').open('rb') as f:_,info=pickle.load(f)
            report=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text()))
            with np.load(d/'perception.npz') as a:occupancy=a['occupancy'].copy();boxes=a['candidate_boxes'].copy();labels=a['candidate_labels'].copy();scores=a['candidate_scores'].copy()
            g,initial=guard(info,report,occupancy,vehicle);gap=float(g.clearance(initial[None],[0.])[0])
            if gap<0:conflicts.append(dict(step=index,proposal_initial_clearance_m=gap))
            if mode=='repair':
                b=output/'baseline'/f'step_{index:03d}';ref=_hugsim_expand(json.loads((b/'scene_reliability.json').read_text()))
                for o in ref['obstacles']:
                    box=np.asarray(o['box'])
                    if o['state']!='network_retained' or not center_visible(box,info):continue
                    result['visible_retained_reference_objects']+=1
                    eligible=(labels==o['label'])&(scores>=ref['detection_threshold'])
                    distance=float(np.linalg.norm(boxes[eligible,:3]-box[:3],axis=1).min()) if eligible.any() else None
                    if distance is None or distance>ref['config']['association_distance_m']:
                        result['visible_retained_reference_misses'].append(dict(step=index,baseline_candidate_index=o['candidate_index'],label=o['label'],nearest_repaired_distance_m=distance))
        result['cases'][mode]=dict(current_proposals_initial_conflict_frames=conflicts,count=len(conflicts),external_driver_memory_replayed=False)
    (output/'screening.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':print(json.dumps(assess(sys.argv[1]),indent=2))
