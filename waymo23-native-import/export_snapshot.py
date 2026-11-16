"""Import every populated Gaussian node at one calibrated frame into HUGSIM.

This is a fixed-view asset validation package, not a dynamic simulation export.
It deliberately cannot be substituted for a full closed-loop scene package.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scene_math import (quat_matrix,matrix_quat,rigid_geometry,smpl_geometry,
    deformation,lighting,pbr_color,sh_eval)

OPENCV_TO_WAYMO=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
WAYMO_TO_HUGSIM=np.linalg.inv(OPENCV_TO_WAYMO)
C0=.28209479177387814


def require(ok,text):
    if not ok:raise ValueError(text)


def camera_metadata(inputs,cfg,state,frame,camera,scale,refine):
    cams=cfg['data']['pixel_source']['cameras']
    require(camera in cams,'camera not in trained camera list')
    cam_index=cams.index(camera)
    start=cfg['data']['start_timestep']
    base=np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt')
    pose=np.loadtxt(inputs/'ego_pose'/f'{start+frame:03d}.txt')
    c2w=np.linalg.inv(base)@pose@np.loadtxt(inputs/'extrinsics'/f'{camera}.txt')@OPENCV_TO_WAYMO
    index=frame*len(cams)+cam_index
    if refine and 'CamPose' in state:
        delta=state['CamPose']['embeds.weight'][index].cpu()
        identity=state['CamPose']['identity'].cpu()
        d=delta[3:]+identity
        a=F.normalize(d[:3],dim=-1)
        b=F.normalize(d[3:]-(a*d[3:]).sum()*a,dim=-1)
        r=torch.stack([a,b,torch.cross(a,b,dim=-1)],dim=-2)
        update=np.eye(4);update[:3,:3]=r.numpy();update[:3,3]=delta[:3].numpy()
        c2w=c2w@update
    # Read dataset metadata from the actual archived training source, without
    # importing that project's optional CUDA/deep-learning dependencies.
    tree=ast.parse((inputs/'backup/datasets/dataset_meta.py').read_text())
    dataset_meta=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATASETS_CONFIG' for t in node.targets):
            dataset_meta=ast.literal_eval(node.value)
    require(dataset_meta is not None,'missing DATASETS_CONFIG')
    original_h,original_w=dataset_meta['waymo'][camera]['original_size']
    down=cfg['data']['pixel_source']['downscale_when_loading'][cam_index]
    h,w=int(original_h/down),int(original_w/down)
    intr=np.loadtxt(inputs/'intrinsics'/f'{camera}.txt')
    k=np.array([[intr[0]*w/original_w,0,intr[2]*w/original_w],
                [0,intr[1]*h/original_h,intr[3]*h/original_h],[0,0,1.]])
    width,height=max(1,round(w*scale)),max(1,round(h*scale))
    k[0]*=width/w;k[1]*=height/h
    return dict(source_c2w=c2w,K=k,width=width,height=height,image_index=index,
                distortion=intr[4:].tolist(),source_frame=start+frame)


def node_geometry(name,s,cfg,frame,frames,step):
    n=len(s['_means'])
    if name in ('Background','RoadNodes'):
        return s['_means'],F.normalize(s['_quats'],dim=-1),torch.ones(n,dtype=torch.bool,device=s['_means'].device)
    if name=='SMPLNodes':return smpl_geometry(s,frame)
    if name=='RigidNodes':return rigid_geometry(s,frame)
    if name=='DeformableNodes':
        ctrl=cfg.get('ctrl',{})
        if ctrl.get('use_deformgs_for_nonrigid',False) and step>ctrl.get('use_deformgs_after',4000):
            delta,dq=deformation(s,cfg['networks'],frame,frames)
            return rigid_geometry(s,frame,delta,dq)
        return rigid_geometry(s,frame)
    raise ValueError('Unsupported nonempty node: '+name)


@torch.no_grad()
def export(checkpoint,inputs,output,frame=0,camera=0,scale=.5,device='cpu',refine=True,remove_rigid=()):
    checkpoint,inputs,output=Path(checkpoint).resolve(),Path(inputs).resolve(),Path(output).resolve()
    require(not output.exists(),'output exists; select a new directory')
    require(0<scale<=1,'scale must be (0,1]')
    cfg=yaml.safe_load((inputs/'config.yaml').read_text())
    require(cfg['data']['dataset']=='waymo','requires Waymo calibration')
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    states=ckpt['models'];step=int(ckpt['step'])
    frames=len(states['RigidNodes']['instances_trans'])
    require(0<=frame<frames,'frame outside checkpoint trajectory')
    for name,state in states.items():
        if '_means' in state and len(state['_means']):
            require(name in ('Background','RoadNodes','RigidNodes','DeformableNodes','SMPLNodes'),
                    'Unsupported populated Gaussian node: '+name)
    for index in remove_rigid:
        require(0<=index<states['RigidNodes']['instances_fv'].shape[1],'bad rigid instance id')
        states['RigidNodes']['instances_fv'][:,index]=False
    cam=camera_metadata(inputs,cfg,states,frame,camera,scale,refine)
    position=torch.tensor(cam['source_c2w'][:3,3],dtype=torch.float32,device=device)
    light=lighting(states['GlobalLighting'],frame,frames,cam['image_index']) if 'GlobalLighting' in states else None
    target=torch.tensor(WAYMO_TO_HUGSIM,dtype=torch.float32,device=device)
    world_q=matrix_quat(target[:3,:3])
    from scene_math import quat_mul
    parts=[];report=[]
    semantic={'Background':2,'RoadNodes':0,'RigidNodes':13,'DeformableNodes':18,'SMPLNodes':11}
    for name,s in states.items():
        if '_means' not in s or not len(s['_means']):continue
        node_cfg=cfg['model'][name]
        xyz,q,valid=node_geometry(name,s,node_cfg,frame,frames,step)
        alpha=torch.sigmoid(s['_opacities']).reshape(-1,1)
        # Invisible objects must not be resurrected when exported as static.
        indices=valid.nonzero().flatten()
        if not len(indices):
            report.append(dict(node=name,total=len(xyz),visible_points=0));continue
        if name=='Background' and node_cfg.get('ctrl',{}).get('use_pbr',False):
            rgb=pbr_color(s,position,light)
            color_mode='training PBR/GlobalLighting equations'
        else:
            coeff=torch.cat([s['_features_dc'][:,None,:],s['_features_rest']],1)
            bases=coeff.shape[1];degree=math.isqrt(bases)-1
            require((degree+1)**2==bases,'non-square SH basis count')
            if degree==0:rgb=torch.sigmoid(s['_features_dc'])
            else:rgb=(sh_eval(coeff,F.normalize(xyz-position,dim=-1))+.5).clamp(0,1)
            color_mode=f'SH degree {degree}'
        raw_scale=s['_scales']
        if raw_scale.shape[-1]==1:raw_scale=raw_scale.expand(-1,3)
        require(raw_scale.shape==xyz.shape,'unsupported Gaussian scales')
        transformed=xyz@target[:3,:3].T+target[:3,3]
        tq=F.normalize(quat_mul(world_q.expand_as(q),q),dim=-1)
        logits=xyz.new_full((len(indices),20),-12);logits[:,semantic[name]]=12
        values=[transformed[indices],((rgb[indices]-.5)/C0)[:,None,:],
                xyz.new_empty((len(indices),0,3)),logits,raw_scale[indices],tq[indices],s['_opacities'][indices].reshape(-1,1)]
        require(all(torch.isfinite(v).all() for v in values),'nonfinite converted values in '+name)
        parts.append([v.detach().cpu() for v in values])
        report.append(dict(node=name,total=len(xyz),visible_points=len(indices),color=color_mode))
        print(name,'exported',len(indices),'/',len(xyz),flush=True)
    require(parts,'no visible Gaussians')
    merged=[torch.cat([p[i] for p in parts],dim=0) for i in range(7)]
    empty=[0]+[v[:0] for v in merged];empty[5]=torch.empty((0,2))
    scene=tuple([0]+merged+[1.0,{},tuple(empty)])
    np.testing.assert_allclose(target.cpu().numpy()[:3,:3].T@target.cpu().numpy()[:3,:3],np.eye(3))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.snapshot-',dir=output.parent))
    try:
        torch.save((scene,step),staging/'scene.pth')
        np.savez(staging/'camera.npz',K=cam['K'],c2w=WAYMO_TO_HUGSIM@cam['source_c2w'],
                 width=cam['width'],height=cam['height'])
        manifest=dict(schema='hugsim.fixed_view_snapshot.v1',checkpoint=str(checkpoint),
            frame=frame,camera=camera,width=cam['width'],height=cam['height'],nodes=report,
            training_camera_refinement=refine,remove_rigid=list(remove_rigid),
            edit_trust=dict(edited=bool(remove_rigid), status="diagnostic_snapshot_only"),
            omitted=['Sky cubemap compositing','Affine postprocessing'],
            restrictions=['fixed frame and fixed camera colors','not a dynamic simulation package',
                          'semantic IDs are coarse node categories; HD-score/DAC not validated'],
            coordinates='Waymo first-ego-relative -> X right, Y down, Z forward',
            source_distortion=cam['distortion'],colors='per-Gaussian RGB baked to SH DC at selected calibrated view')
        (staging/'snapshot.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging);raise
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frame',type=int,default=0)
    p.add_argument('--camera',type=int,default=0);p.add_argument('--scale',type=float,default=.5)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--no-camera-refinement',action='store_true')
    p.add_argument('--remove-rigid',type=int,nargs='*',default=[])
    a=p.parse_args()
    result=export(a.checkpoint,a.inputs,a.output,a.frame,a.camera,a.scale,a.device,
                  not a.no_camera_refinement,a.remove_rigid)
    print(json.dumps(result,indent=2,ensure_ascii=False))
