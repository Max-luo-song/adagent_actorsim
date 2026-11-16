"""Post-hoc source-annotation occupancy audit; never supplies labels to inference."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from paired_edit_diagnostic import asset_corners
from evaluate_preview_vehicle import polygon_iou


def audit(sequence_path,asset_path):
    sequence=json.loads(Path(sequence_path).read_text());source=json.loads(Path(sequence['source_report']).read_text())
    data=Path(source.get('data_dir',source.get('original_data',source['inputs'])));inputs=Path(source['inputs']);start=int(source.get('start_timestep',0))
    origin=np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt');annotations=json.loads((data/'instances/instances_info.json').read_text())
    corners=asset_corners(json.loads(Path(asset_path).read_text()));hull=ConvexHull(corners[:,:2]);polygon=corners[hull.vertices,:2];records=[]
    for frame in sequence['frames']:
        raw_frame=start+frame['frame'];hits=[]
        for identity,actor in annotations.items():
            ann=actor['frame_annotations']
            if raw_frame not in ann['frame_idx']:continue
            j=ann['frame_idx'].index(raw_frame);size=np.asarray(ann['box_size'][j]);pose=np.linalg.inv(origin)@np.asarray(ann['obj_to_world'][j])
            local=np.array([[x,y,z,1.] for x in (-size[0]/2,size[0]/2) for y in (-size[1]/2,size[1]/2) for z in (-size[2]/2,size[2]/2)])
            world=local@pose.T
            vertical=min(corners[:,2].max(),world[:,2].max())-max(corners[:,2].min(),world[:,2].min())
            if vertical<=0:continue
            actor_hull=ConvexHull(world[:,:2]);iou=polygon_iou(polygon,world[actor_hull.vertices,:2])
            if iou>1e-9:hits.append(dict(instance_key=identity,class_name=actor['class_name'],bev_iou=float(iou),z_interval_overlap_m=float(vertical)))
        records.append(dict(frame=frame['frame'],possible_body_intersections=hits))
    return dict(records=records,source_annotations_used_only_offline=True,
                interpretation='Positive XY hull intersection and Z interval overlap; conservative proxy, not exact mesh collision',
                n1_edit_certified=False)


if __name__=='__main__':
    import sys
    root=Path('outputs/edit_vehicle_resolution_20261021');result={}
    for scene,n in [(172,10),(552,20)]:
        folder=Path(f'outputs/gpu_test_service/jobs/edfix-{n}-{scene}-render-v1/result')
        result[str(scene)]=audit(folder/'sequence_report.json',folder/'asset_report.json')
    folder=Path('outputs/gpu_test_service/jobs/edwarm-10-552-render-v1/result');result['552_warmup']=audit(folder/'sequence_report.json',folder/'asset_report.json')
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
