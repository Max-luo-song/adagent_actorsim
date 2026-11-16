"""Post-hoc source-scene audit; simulator truth never enters control."""

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

import hashlib
import json
pass  # Path is provided by the portable boundary above
import pickle
import sys
import numpy as np
import torch
import yaml

P28=Path(__file__).resolve().parents[1]
ROOT=P28.parents[5]
sys.path[:0]=[str(ROOT/'waymo23-native-import'),str(ROOT/'integrations/sr-hugsim'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from scene_math import rigid_geometry,smpl_geometry,deformation
from native_actor_evaluation import NativeActorEvaluation
from box_geometry import box_corners
from input_geometry import camera_projection

OUT=P28/'runs/near_ego_origin_audit_20261102'
CHECKPOINT=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'))

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'scene_existence.json').exists(), 'Refusing existing results'
    step=P28/'runs/heading_fix_scene552_v1/uniad_worker/step_039'
    with open(step/'sensor_metadata.pkl','rb') as f:info=pickle.load(f)
    with np.load(step/'perception.npz') as d:box=d['candidate_boxes'][33].astype(float);refined=d['geometry_candidate_boxes'][33].copy();gravity=d['candidate_boxes_gravity'][33].copy()
    rotation=np.asarray(info['explicit_l2g_r_mat']);translation=np.asarray(info['explicit_l2g_t'])
    # source (forward,left,up) -> model-global (-left,forward,up)
    S=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    center=S.T@(rotation@box[:3]+translation)
    corners=(box_corners(box)@rotation.T+translation)@S
    actors=NativeActorEvaluation.from_archive(dict(np.load(P28.parent/'phase5/native552_actor_geometry.npz')))
    actor_rows=[]
    for group,index,poly,z0,z1 in actors.geometry(119.):
        mid=poly.mean(0);dist=float(np.linalg.norm(mid-center[:2]))
        actor_rows.append(dict(group=group,index=index,center_distance_m=dist,
            enclosing_disk_gap_lower_bound_m=dist-float(np.linalg.norm(poly-mid,axis=1).max())-float(np.linalg.norm(corners[:,:2]-center[:2],axis=1).max()),
            center=mid.tolist(),height=[z0,z1]))
    cfg=yaml.safe_load((CHECKPOINT.parent/'config.yaml').read_text())
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    models=checkpoint['models'];geometry=[]
    # Every model's rendered Gaussian center is compared in the ORIGINAL
    # candidate local box, not a mismatched global coordinate system.
    for name,state in models.items():
        if '_means' not in state:continue
        if name in ('RigidNodes','TrafficLightNodes','DeformableNodes'):
            if name=='DeformableNodes':
                delta,dq=deformation(state,cfg['model'][name]['networks'],119,198)
                xyz,_,valid=rigid_geometry(state,119,delta,dq)
            else:xyz,_,valid=rigid_geometry(state,119)
            xyz=xyz[valid]
        elif name=='SMPLNodes':
            xyz,_,valid=smpl_geometry(state,119);xyz=xyz[valid]
        else:xyz=state['_means']
        xyz=xyz.detach().numpy().astype(float)
        # source->model-global->model-local
        local=(xyz@S.T-translation)@rotation
        delta=local-box[:3];cs,sn=np.cos(box[6]),np.sin(box[6])
        flat=delta[:,:2]@np.array([[cs,sn],[-sn,cs]])
        in_xy=(np.abs(flat)<=box[3:5]/2).all(1)
        in_z=(delta[:,2]>=0)&(delta[:,2]<=box[5])
        in_body=in_xy&in_z
        row=dict(group=name,centers=len(xyz),centers_in_reported_box=int(in_body.sum()),
            centers_in_xy_footprint=int(in_xy.sum()),centers_in_box_above_source_z_0p5=int((in_body&(xyz[:,2]>.5)).sum()),
            minimum_center_distance_xy_m=float(np.linalg.norm(xyz[:,:2]-center[:2],axis=1).min()),
            footprint_source_z_quantiles=np.quantile(xyz[in_xy,2],[0,.1,.5,.9,1]).tolist() if in_xy.any() else None)
        geometry.append(row);print(row,flush=True)
    inp=_hugsim_expand(json.loads((step/'input_geometry.json').read_text()))
    projection=[]
    for index,name in enumerate(['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT']):
        camera=info['cam_params'][name]
        for mode in ('stretch','letterbox'):
            k,p=camera_projection(camera,(camera['intrinsic']['H'],camera['intrinsic']['W'],3),resize_mode=mode)
            projection.append(dict(camera=name,assumed_resize_mode=mode,intrinsic_max_error=float(np.abs(k-np.asarray(inp['intrinsic_matrices'][index])).max()),
                projection_max_error=float(np.abs(p-np.asarray(inp['lidar_to_image_matrices'][index])).max())))
    output=dict(diagnostic_uncertified=True,formal_evaluation_admitted=False,
        candidate_source_bottom_center=center.tolist(),source_frame=119,
        raw_equals_refined=bool(np.array_equal(box,refined)),
        gravity_to_bottom_xy_error_m=float(np.abs(gravity[:2]-box[:2]).max()),
        gravity_to_bottom_z_error_m=float(abs(gravity[2]-box[2]-box[5]/2)),
        actors=sorted(actor_rows,key=lambda r:r['center_distance_m']),gaussian_geometry=geometry,
        actual_input_projection=projection,
        scope='Saved scene checkpoint and active Gaussian centers, no rendering or neural inference. Gaussian centers are not mesh volumes or certified physical truth. Inserted asset evaluated separately by existing placement audit.',
        source_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [step/'sensor_metadata.pkl',step/'perception.npz',step/'input_geometry.json',CHECKPOINT.parent/'config.yaml']})
    (OUT/'scene_existence.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
