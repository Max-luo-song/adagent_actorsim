"""Explain candidate visibility and memory admission from saved inputs."""

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
import sys
import numpy as np

P28=Path(__file__).resolve().parents[1]
ROOT=P28.parents[5]
sys.path.insert(0,str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import box_corners
RUN=P28/'runs/heading_fix_scene552_v1/uniad_worker'
OUT=P28/'runs/near_ego_causal_audit_20261102'

def main():
    report=_hugsim_expand(json.loads((RUN/'step_039/scene_reliability.json').read_text()))
    candidate=next(o for o in report['obstacles'] if o['candidate_index']==33)
    geometry=_hugsim_expand(json.loads((RUN/'step_039/input_geometry.json').read_text()))
    box=np.asarray(candidate['box'])
    points=np.vstack([box[:3]+[0,0,box[5]/2],box_corners(box)])
    points=np.c_[points,np.ones(9)]
    visibility={}
    for name,matrix,shape in zip(['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'],geometry['lidar_to_image_matrices'],geometry['image_shapes']):
        q=points@np.asarray(matrix).T
        z=q[:,2];uv=q[:,:2]/np.maximum(z[:,None],1e-6)
        valid=(z>.001)&(uv[:,0]>=0)&(uv[:,0]<shape[1])&(uv[:,1]>=0)&(uv[:,1]<shape[0])
        visibility[name]=dict(visible_point_indices=np.flatnonzero(valid).tolist(),
            visible_pixels=uv[valid].tolist(),center_camera_depth_m=float(z[0]),
            index_convention='0=gravity center;1..8=box corners',occlusion_or_depth_test=False)
    memory=[]
    for step in range(39,69):
        rel=_hugsim_expand(json.loads((RUN/f'step_{step:03d}/guard_memory_report.json').read_text()))
        rows=[o for o in rel['obstacle_memory'].get('constraints',[]) if o.get('memory_id')==88]
        # Current matched detections replace the memory row in final obstacles.
        emitted=[o for o in rel['obstacles'] if o.get('memory_id')==88]
        memory.append(dict(step=step,memory_constraint=rows,emitted_memory=emitted,
            nearby_current=[o for o in rel['obstacles'] if o.get('label')==0 and o['candidate_index']>=0 and np.linalg.norm(o['box'][:2])<2]))
    output=dict(candidate=candidate,visibility=visibility,memory88_history=memory,
        mechanism='Any projected box corner in a camera is sufficient for visible=True; no surface validation. Score>=0.4 directly retains the single observation; guard hardens it. Later confirmed memory persists after missing detections.',
        caveat='This explains admission and persistence, not which internal neural feature generated the proposal. Candidate indices and evidence_track_id are not saved neural query identities.')
    (OUT/'admission_evidence.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(visibility,indent=2))

if __name__=='__main__':main()
