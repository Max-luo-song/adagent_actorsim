"""Use the installed HUGSIM renderer to verify the imported fixed-view assets."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch


def to_cuda(x):
    if torch.is_tensor(x):return x.to('cuda')
    if isinstance(x,tuple):return tuple(to_cuda(v) for v in x)
    if isinstance(x,list):return [to_cuda(v) for v in x]
    if isinstance(x,dict):return {k:to_cuda(v) for k,v in x.items()}
    return x


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hugsim',required=True);p.add_argument('--package',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    repo,package,output=map(lambda x:Path(x).resolve(),(a.hugsim,a.package,a.output))
    if not torch.cuda.is_available():raise RuntimeError('Run in existing HUGSIM CUDA environment')
    if output.exists():raise ValueError('Use a new output directory')
    metadata=json.loads((package/'snapshot.json').read_text())
    if metadata['schema']!='hugsim.fixed_view_snapshot.v1':raise ValueError('wrong package format')
    sys.path.insert(0,str(repo));os.chdir(repo)
    import gaussian_renderer as renderer
    from scene.cameras import Camera
    from PIL import Image
    values,iteration=torch.load(package/'scene.pth',map_location='cpu',weights_only=False)
    if len(values)!=11:raise ValueError('Expected 11-field native scene tuple')
    pc=renderer.GaussianModel(0,affine=True)
    pc.restore(to_cuda(values),None)
    pc.affine=False
    if len(pc.get_full_xyz)!=sum(n['visible_points'] for n in metadata['nodes']):
        raise ValueError('Point count mismatch after HUGSIM restore')
    c=np.load(package/'camera.npz',allow_pickle=False)
    width,height=int(c['width']),int(c['height'])
    k=np.eye(4);k[:3,:3]=c['K']
    camera=Camera(width=width,height=height,image=np.zeros((height,width,3)),
                  K=k,c2w=c['c2w'],image_name='imported',dynamics={})
    original=renderer.concatenate_all
    renderer.concatenate_all=lambda fg:original(fg) if fg else [None]*6
    result=renderer.render(viewpoint=camera,prev_viewpoint=None,pc=pc,dynamic_gaussians={},
        unicycles={},bg_color=torch.zeros(3,device='cuda'),planning=[])
    rgb=result['render'].permute(1,2,0).clamp(0,1).cpu().numpy()
    depth=result['depth'][0].cpu().numpy()
    if rgb.shape!=(height,width,3) or not np.isfinite(rgb).all() or not np.isfinite(depth).all():
        raise ValueError('Invalid output shape or nonfinite render')
    output.mkdir(parents=True)
    Image.fromarray((rgb*255).astype(np.uint8)).save(output/'rgb.png')
    np.save(output/'depth.npy',depth)
    alpha=result['alphas'].detach().cpu().numpy()
    np.save(output/'alpha.npy',alpha)
    info=dict(status='native HUGSIM asset restore and single-view rasterization passed',
        gpu=torch.cuda.get_device_name(),points=len(pc.get_full_xyz),width=width,height=height,
        rgb_min=float(rgb.min()),rgb_max=float(rgb.max()),rgb_std=float(rgb.std()),
        alpha_mean=float(alpha.mean()),closed_loop_tested=False,quality_verified=False)
    (output/'render_report.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2));print('image:',output/'rgb.png')


if __name__=='__main__':main()
