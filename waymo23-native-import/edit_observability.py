"""Geometric truncation diagnostics; does not infer visibility behind occluders."""
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from evaluate_preview_vehicle import polygon_iou
from paired_edit_diagnostic import asset_corners


def projected_extent(corners,camera_to_world,intrinsic,image_wh):
    points=np.column_stack([np.asarray(corners),np.ones(len(corners))])@np.linalg.inv(camera_to_world).T
    if np.any(points[:,2]<=.1):
        return dict(projectable=False,reason='Bounds cross/behind near camera plane',occlusion_verified=False)
    pixels=points[:,:3]@np.asarray(intrinsic).T;uv=pixels[:,:2]/pixels[:,2:3]
    hull=ConvexHull(uv);polygon=uv[hull.vertices];area=hull.volume
    width,height=image_wh;rectangle=np.array([[0,0],[width,0],[width,height],[0,height]])
    iou=polygon_iou(polygon,rectangle)
    intersection=iou*(area+width*height)/(1+iou)
    return dict(projectable=True,bounds_uv=[uv.min(0).tolist(),uv.max(0).tolist()],
                projected_hull_area_pixels=float(area),in_image_projected_fraction=float(np.clip(intersection/area,0,1)),
                truncated=bool((uv[:,0]<0).any() or (uv[:,0]>width).any() or (uv[:,1]<0).any() or (uv[:,1]>height).any()),
                occlusion_verified=False)


def audit(sequence_path,asset_path):
    sequence=json.loads(Path(sequence_path).read_text());asset=json.loads(Path(asset_path).read_text());corners=asset_corners(asset);rows=[]
    for frame in sequence['frames']:
        packet=Path(frame['packet']);packet=packet if packet.is_absolute() else Path(sequence_path).parent/packet
        with packet.open('rb') as stream:obs,_=pickle.load(stream)
        with np.load(packet.parent/'render_evidence.npz',allow_pickle=False) as views:
            for name,image in obs['rgb'].items():
                result=projected_extent(corners,views[name+'__camera_to_world'],views[name+'__K'],(image.shape[1],image.shape[0]))
                rows.append(dict(frame=frame['frame'],camera=name,**result))
    return dict(records=rows,reference_is_placement_proxy=True,full_visibility_certified=False,n1_edit_certified=False)


if __name__=='__main__':
    import sys
    results={}
    for scene in (172,552):
        pair=json.loads(Path(f'outputs/box_generalization_fix_20261020/scene_{scene}_pair.json').read_text())
        results[str(scene)]=audit(pair['edited_sequence'],pair['asset_report'])
    Path(sys.argv[1]).write_text(json.dumps(results,indent=2)+'\n')
