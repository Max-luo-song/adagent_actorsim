"""Audit asset SH covariance on saved physical camera directions, without rendering."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from asset_radiance import evaluator,sh_rotation


def audit(sequence_path,asset_path):
    sequence=json.loads(Path(sequence_path).read_text());asset=json.loads(Path(asset_path).read_text())
    params,_=torch.load(Path(asset['asset'])/'gs.pth',map_location='cpu',weights_only=False)
    degree,xyz,dc,rest,_,_,_,opacity,_=params
    solid=np.flatnonzero((torch.sigmoid(opacity).detach().numpy().ravel()>.5))
    indices=solid[np.linspace(0,len(solid)-1,min(2048,len(solid)),dtype=int)]
    points=xyz.detach().numpy()[indices].copy();points[:,[0,2]]-=asset['local_recenter_xz']
    coefficients=torch.cat((dc,rest),dim=1).detach().numpy()[indices].astype(float)
    pose=np.asarray(asset['placements'][0]['asset_to_native']);source_pose=np.asarray(asset['placements'][0]['asset_to_source'])
    source_to_native=pose@np.linalg.inv(source_pose)
    world=points@pose[:3,:3].T+pose[:3,3];matrix=sh_rotation(pose[:3,:3],int(degree))
    corrected=np.einsum('ij,njc->nic',matrix,coefficients);rows=[]
    for frame in sequence['frames']:
        packet=Path(frame['packet']);packet=packet if packet.is_absolute() else Path(sequence_path).parent/packet
        with np.load(packet.parent/'render_evidence.npz',allow_pickle=False) as views:
            with packet.open('rb') as stream:obs,_=pickle.load(stream)
            for name in obs['rgb']:
                c2s=views[name+'__camera_to_world'];cam=(source_to_native@c2s)[:3,3]
                direction=world-cam;direction/=np.linalg.norm(direction,axis=1,keepdims=True)
                expected=evaluator()(int(degree),coefficients.transpose(0,2,1),direction@pose[:3,:3])
                before=evaluator()(int(degree),coefficients.transpose(0,2,1),direction)
                after=evaluator()(int(degree),corrected.transpose(0,2,1),direction)
                rows.append(dict(frame=frame['frame'],camera=name,
                    before_mean_absolute_radiance_error=float(np.abs(before-expected).mean()),
                    after_max_absolute_radiance_error=float(np.abs(after-expected).max()),
                    before_after_mean_clipped_rgb_difference=float(np.abs(np.clip(before+.5,0,1)-np.clip(after+.5,0,1)).mean())))
    return dict(sampled_opaque_gaussians=len(indices),records=rows,
                reference='Local SH evaluation at physically corresponding rotated directions; not an independent photograph',
                photorealism_certified=False,n1_edit_certified=False)


if __name__=='__main__':
    output={}
    for scene in (172,552):
        pair=json.loads(Path(f'outputs/box_generalization_fix_20261020/scene_{scene}_pair.json').read_text())
        output[str(scene)]=audit(pair['edited_sequence'],pair['asset_report'])
    Path(sys.argv[1]).write_text(json.dumps(output,indent=2)+'\n')
