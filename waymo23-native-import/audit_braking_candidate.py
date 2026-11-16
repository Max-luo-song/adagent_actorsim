"""Offline audit only: annotations never enter inference or control."""
import json
from pathlib import Path
import sys
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'))
from box_geometry import box_corners


def audit():
    root=BASE/'outputs/gpu_test_service/jobs/bevsplit-172-20261021-v1/result/history'
    contributor=json.loads((BASE/'outputs/mpc_braking_diagnosis_20261021/replay_step002.json').read_text())['contributor']
    box=np.asarray(contributor['box']);meta=json.loads((root/'step_002/input_geometry.json').read_text())
    corners=box_corners(box);points=np.c_[corners,np.ones(8)];views=[]
    for name,p in zip(('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'),meta['lidar_to_image_matrices']):
        q=points@np.asarray(p).T;uv=q[:,:2]/np.where(np.abs(q[:,2:3])>1e-8,q[:,2:3],np.nan)
        valid=(q[:,2]>1e-3)&(uv[:,0]>=0)&(uv[:,0]<1600)&(uv[:,1]>=0)&(uv[:,1]<900)
        padded=(q[:,2]>1e-3)&(uv[:,0]>=0)&(uv[:,0]<1600)&(uv[:,1]>=900)&(uv[:,1]<928)
        views.append(dict(camera=name,corners_in_real_image=int(valid.sum()),corners_only_in_padding=int(padded.sum()),
                          corner_uv=uv.tolist(),corner_depth=q[:,2].tolist(),occlusion_verified=False))
    rot=np.asarray(meta['l2g_rotation_row']).T;trans=np.asarray(meta['l2g_translation'])
    world=rot@box[:3]+trans
    temporal=[]
    for i in range(30):
        folder=root/f'step_{i:03d}';r=json.loads((folder/'scene_reliability.json').read_text());m=json.loads((folder/'input_geometry.json').read_text())
        detections=np.load(folder/'perception.npz');b=detections['candidate_boxes'];scores=detections['candidate_scores'];labels=detections['candidate_labels']
        w=b[:,:3]@np.asarray(m['l2g_rotation_row'])+np.asarray(m['l2g_translation']);dist=np.linalg.norm(w-world,axis=1);dist=np.where(labels==0,dist,np.inf);j=int(np.argmin(dist))
        track=[o for o in r['obstacles'] if o['evidence_track_id']==contributor['evidence_track_id']]
        temporal.append(dict(step=i,nearest_vehicle_candidate_distance_m=float(dist[j]),nearest_score=float(scores[j]),
                             nearest_box=b[j].tolist(),same_guard_track=track))
    source=json.loads((BASE/'outputs/waymo172_three_setup_20261016_132050_295391/source_report.json').read_text())
    data=Path(source['original_data']);annotations=data/'instances/instances_info.json';native=[];native_error=None
    try:
        ann=json.loads(annotations.read_text());origin=np.loadtxt(sorted((Path(source['inputs'])/'ego_pose').glob('*.txt'))[0]);A=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        center_source=A.T@(world+rot@np.array([0.,0.,box[5]/2]))
        for key,item in ann.items():
            if item['class_name']!='Vehicle':continue
            frames=item['frame_annotations'];frame=148+int(source.get('start_timestep',0))
            if frame not in frames['frame_idx']:continue
            j=frames['frame_idx'].index(frame);pose=np.linalg.inv(origin)@np.asarray(frames['obj_to_world'][j]);native.append(dict(identity=key,distance_3d_m=float(np.linalg.norm(pose[:3,3]-center_source)),center_source=pose[:3,3].tolist(),box_lwh=frames['box_size'][j]))
        native.sort(key=lambda r:r['distance_3d_m'])
    except (OSError,ValueError,KeyError) as exc:native_error=repr(exc)
    out=dict(candidate=contributor,projection=views,temporal=temporal,nearest_native_annotations=native[:5],native_annotation_error=native_error,
             raw_frame_assumption=148,annotation_used_only_offline=True,ground_truth_to_model=False,
             source_render_domain_difference=True,authenticity_independently_proven=False,generalization_proven=False)
    target=BASE/'outputs/candidate_authenticity_20261021/audit.json';target.write_text(json.dumps(out,indent=2))
    print(json.dumps(dict(views=[{k:v for k,v in r.items() if k in ('camera','corners_in_real_image','corners_only_in_padding')} for r in views],native=native[:3],error=native_error,track_steps=[r['step'] for r in temporal if r['same_guard_track']]),indent=2))

if __name__=='__main__':audit()
