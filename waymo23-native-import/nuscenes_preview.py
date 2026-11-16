"""Native HUGSIM preview using six real nuScenes camera poses; no vehicle loop."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from export_snapshot import node_geometry, C0
from scene_math import rigid_geometry, pbr_color, lighting, sh_eval

NAMES = {0:'CAM_FRONT',1:'CAM_FRONT_LEFT',2:'CAM_FRONT_RIGHT',
         3:'CAM_BACK_LEFT',4:'CAM_BACK_RIGHT',5:'CAM_BACK'}


def archived_color_network(path, networks, state):
    text = path.read_text()
    wanted = {'Embedder','get_embedder','ColorConditionalDeformNetwork'}
    nodes = [n for n in ast.parse(text).body
             if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('archived traffic network dependencies differ')
    namespace = dict(torch=torch,nn=nn,F=F,np=np,math=math)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    net = namespace['ColorConditionalDeformNetwork'](input_ch=3,**networks)
    weights = {k[len('deform_network.'):]:v for k,v in state.items() if k.startswith('deform_network.')}
    net.load_state_dict(weights,strict=True)
    return net.to(state['_means'].device).eval()


@torch.no_grad()
def traffic_geometry(state, config, frame, frames, step, network):
    ctrl = config['ctrl']
    dx = dq = ds = dc = None
    if ctrl.get('use_deformgs_for_nonrigid',False) and step>ctrl.get('use_deformgs_after',4000):
        ids = state['points_ids'].flatten().long()
        height = state['instances_size'][ids,2,None]
        if (height<=0).any():
            raise ValueError('invalid traffic instance height')
        x = state['_means']/height*2
        t = x.new_full((len(x),1),frame/max(1,frames-1))
        dx,dq,ds,dc = network(x,t,state['instances_embedding'][ids])
    xyz,quat,valid = rigid_geometry(state,frame,dx,dq)
    scale = state['_scales']
    if scale.shape[-1]==1:
        scale = scale.expand(-1,3)
    if ds is not None:
        activated = scale.exp()+ds
        if (activated<=0).any():
            raise ValueError('nonpositive deformed traffic scale')
        scale = activated.log()
    coeff = torch.cat([state['_features_dc'][:,None,:],state['_features_rest']],dim=1)
    if dc is not None:
        coeff = coeff+dc.reshape_as(coeff)
    return xyz,quat,valid,scale,coeff


def camera_metadata(data, config, frame, camera, scale):
    cams = config['data']['pixel_source']['cameras']
    start = int(config['data']['start_timestep'])
    first = np.loadtxt(data/'extrinsics'/f'{start:03d}_0.txt')
    current = np.loadtxt(data/'extrinsics'/f'{start+frame:03d}_{camera}.txt')
    pose = np.linalg.inv(first)@current
    down = config['data']['pixel_source']['downscale_when_loading'][cams.index(camera)]
    w,h = max(1,round(int(1600/down)*scale)),max(1,round(int(900/down)*scale))
    intr = np.loadtxt(data/'intrinsics'/f'{camera}.txt')
    if np.any(np.abs(intr[4:])>1e-10):
        raise ValueError('nonzero distortion requires explicit image-model handling')
    k = np.array([[intr[0]*w/1600,0,intr[2]*w/1600],
                  [0,intr[1]*h/900,intr[3]*h/900],[0,0,1.]])
    if not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
        raise ValueError('invalid camera transform')
    return dict(c2w=pose,K=k,width=w,height=h)


@torch.no_grad()
def main():
    from PIL import Image,ImageDraw
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--scale',type=float,default=1.)
    args = parser.parse_args()
    if not 0<args.scale<=1:
        raise ValueError('scale must be in (0,1]')
    base = Path(os.environ['BASE'])
    shared = Path('/inspire/hdd/global_user/guoluosong-253108120129/adscenegraph')
    scene = shared/'Omniscenegraph/tmp/outputs/nuscene/scene-0001'
    data = shared/'data/nuscenes/processed_10Hzads/mini/001'
    config = yaml.safe_load((scene/'config.yaml').read_text())
    if config['data']['dataset']!='nuscenes' or config['data']['pixel_source']['cameras']!=list(range(6)):
        raise ValueError('unexpected dataset or camera order')
    if config['data']['pixel_source'].get('undistort',False):
        raise ValueError('undistortion path not implemented in this preview')
    repo = base/'code/HUGSIM'
    sys.path[:0] = [str(repo),str(repo/'sim')]
    os.chdir(repo)
    import gaussian_renderer as renderer
    from scene.cameras import Camera
    out = base/'outputs'/('nuscenes0001_preview_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print('OUTPUT:',out,flush=True)
    checkpoint = scene/'checkpoint_final.pth'
    print('Loading scene checkpoint once...',flush=True)
    ckpt = torch.load(checkpoint,map_location='cuda',weights_only=False)
    states,step = ckpt['models'],int(ckpt['step'])
    frames = len(states['RigidNodes']['instances_trans'])
    if not 0<=args.frame<frames:
        raise ValueError('frame outside checkpoint range')
    labels = dict(Background=2,RoadNodes=0,RigidNodes=13,DeformableNodes=18,SMPLNodes=11,TrafficLightNodes=6)
    parts = [];counts = {}
    modules = scene/'backup/models/modules.py'
    for name,state in states.items():
        if '_means' not in state or not len(state['_means']):
            continue
        if name not in labels:
            raise ValueError('unsupported populated node: '+name)
        coeff = None
        if name=='TrafficLightNodes':
            network = archived_color_network(modules,config['model'][name]['networks'],state)
            xyz,q,valid,scales,coeff = traffic_geometry(state,config['model'][name],args.frame,frames,step,network)
        else:
            xyz,q,valid = node_geometry(name,state,config['model'][name],args.frame,frames,step)
            scales = state['_scales']
            if scales.shape[-1]==1:
                scales = scales.expand(-1,3)
        idx = valid.nonzero().flatten()
        counts[name] = dict(total=len(state['_means']),included=len(idx))
        if not len(idx):
            continue
        logits = xyz.new_full((len(idx),20),-12);logits[:,labels[name]]=12
        parts.append(dict(name=name,idx=idx,xyz=xyz[idx],q=q[idx],scale=scales[idx],
                          alpha=state['_opacities'][idx].reshape(-1,1),logits=logits,
                          coeff=None if coeff is None else coeff[idx]))
        print(name,counts[name],flush=True)
    pc = renderer.GaussianModel(0,affine=False)
    for field,key in [('_xyz','xyz'),('_rotation','q'),('_scaling','scale'),('_opacity','alpha'),('_feats3D','logits')]:
        setattr(pc,field,torch.cat([part[key] for part in parts]))
    pc._features_rest = pc._xyz.new_empty((len(pc._xyz),0,3))
    original = renderer.concatenate_all
    renderer.concatenate_all = lambda fg:original(fg) if fg else [None]*6
    images = {};cameras = {}
    try:
        for cid,name in NAMES.items():
            c = camera_metadata(data,config,args.frame,cid,args.scale)
            position = pc._xyz.new_tensor(c['c2w'][:3,3])
            light = lighting(states['GlobalLighting'],args.frame,frames,args.frame*6+cid)
            colors = []
            for part in parts:
                state = states[part['name']];idx = part['idx']
                if part['name']=='Background' and config['model']['Background'].get('ctrl',{}).get('use_pbr',False):
                    rgb = pbr_color(state,position,light)[idx]
                else:
                    coeff = part['coeff']
                    if coeff is None:
                        coeff = torch.cat([state['_features_dc'][idx,None,:],state['_features_rest'][idx]],1)
                    degree = min(step//config['trainer']['gaussian_ctrl_general_cfg']['sh_degree_interval'],
                                 int(round(math.sqrt(coeff.shape[1])))-1)
                    rgb = (sh_eval(coeff[:,:((degree+1)**2)],F.normalize(part['xyz']-position,dim=-1))+.5).clamp(0,1) if coeff.shape[1]>1 else torch.sigmoid(coeff[:,0])
                colors.append(((rgb-.5)/C0)[:,None,:])
            pc._features_dc = torch.cat(colors)
            k = np.eye(4);k[:3,:3] = c['K']
            camera = Camera(width=c['width'],height=c['height'],
                            image=np.zeros((c['height'],c['width'],3)),K=k,c2w=c['c2w'],
                            image_name=name,dynamics={})
            result = renderer.render(viewpoint=camera,prev_viewpoint=None,pc=pc,dynamic_gaussians={},
                                     unicycles={},bg_color=torch.zeros(3,device='cuda'),planning=[])
            rgb = result['render'].permute(1,2,0).clamp(0,1).cpu().numpy()
            alpha = result['alphas'].cpu().numpy()
            if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
                raise RuntimeError('nonfinite render')
            image = (rgb*255).astype(np.uint8)
            images[name] = Image.fromarray(image);images[name].save(out/(name+'.png'))
            cameras[name] = dict(width=c['width'],height=c['height'],K=c['K'].tolist(),
                                 c2w=c['c2w'].tolist(),virtual=False,
                                 nonblack_fraction=float((image.max(2)>3).mean()),
                                 alpha_coverage_fraction=float((alpha>.1).mean()))
            print(name,'nonblack:',round(cameras[name]['nonblack_fraction'],3),
                  'alpha coverage:',round(cameras[name]['alpha_coverage_fraction'],3),flush=True)
    finally:
        renderer.concatenate_all = original
    order = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
    mosaic = Image.new('RGB',(1440,588),'white');draw = ImageDraw.Draw(mosaic)
    for i,name in enumerate(order):
        x,y=i%3*480,i//3*294
        draw.text((x+5,y+5),name,fill='black')
        mosaic.paste(images[name].resize((480,270)),(x,y+24))
    mosaic.save(out/'six_views.png')
    report = dict(stage='six real-camera native HUGSIM preview',checkpoint=str(checkpoint),
                  data=str(data),frame=args.frame,frames=frames,nodes=counts,cameras=cameras,
                  coordinates='relative to first front camera; OpenCV axes',
                  traffic_network='archived class with strict weight loading',
                  archived_modules_sha256=hashlib.sha256(modules.read_bytes()).hexdigest(),
                  sky_and_affine_enabled=False,camera_refinement=False,
                  closed_loop_tested=False,quality_verified=False)
    (out/'preview_report.json').write_text(json.dumps(report,indent=2))
    print('OUTPUT:',out)


if __name__=='__main__':
    main()
