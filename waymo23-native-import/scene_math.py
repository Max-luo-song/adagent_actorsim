"""Small Torch-only evaluators for the supplied 2026-04-18 training snapshot."""
import math
import torch
import torch.nn.functional as F


def quat_matrix(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))


def quat_mul(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],-1)


def matrix_quat(m):
    # Trace-based candidates remain stable near 180 degrees. The source also
    # converts blended LBS matrices this way before normalizing the quaternion.
    m00,m01,m02=m[...,0,:].unbind(-1)
    m10,m11,m12=m[...,1,:].unbind(-1)
    m20,m21,m22=m[...,2,:].unbind(-1)
    a=torch.sqrt(torch.clamp(torch.stack([1+m00+m11+m22,1+m00-m11-m22,
        1-m00+m11-m22,1-m00-m11+m22],-1),min=0))
    c=torch.stack([torch.stack([a[...,0]**2,m21-m12,m02-m20,m10-m01],-1),
        torch.stack([m21-m12,a[...,1]**2,m10+m01,m02+m20],-1),
        torch.stack([m02-m20,m10+m01,a[...,2]**2,m12+m21],-1),
        torch.stack([m10-m01,m20+m02,m21+m12,a[...,3]**2],-1)],-2)
    c=c/(2*a.clamp_min(.1)[...,None])
    idx=a.argmax(-1)
    return F.normalize(c.gather(-2,idx[...,None,None].expand(idx.shape+(1,4))).squeeze(-2),dim=-1)


def sh_basis(d,degree):
    x,y,z=d.unbind(-1);one=torch.ones_like(x)
    b=[one*.28209479177387814]
    if degree>=1:b += [-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x]
    if degree>=2:b += [1.0925484305920792*x*y,-1.0925484305920792*y*z,
        .31539156525252005*(2*z*z-x*x-y*y),-1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)]
    if degree>=3:b += [-.5900435899266435*y*(3*x*x-y*y),2.890611442640554*x*y*z,
        -.4570457994644658*y*(4*z*z-x*x-y*y),.3731763325901154*z*(2*z*z-3*x*x-3*y*y),
        -.4570457994644658*x*(4*z*z-x*x-y*y),1.445305721320277*z*(x*x-y*y),
        -.5900435899266435*x*(x*x-3*y*y)]
    if degree not in range(4):raise ValueError('SH degree outside 0..3')
    return torch.stack(b,-1)


def sh_eval(coeff,d):
    degree=int(math.isqrt(coeff.shape[-2]))-1
    return (sh_basis(d,degree)[...,None]*coeff).sum(-2)


def embed(x,multires):
    return torch.cat([x]+[fn(x*(2.**i)) for i in range(multires) for fn in (torch.sin,torch.cos)],-1)


def deformation(state,cfg,frame,frames,chunk=8192):
    ids=state['points_ids'].flatten().long()
    means=state['_means'];sizes=state['instances_size']
    output=[];rotations=[]
    if cfg.get('deform_scale',False):raise ValueError('deform_scale not supported by this verified snapshot')
    for offset in range(0,len(means),chunk):
        sl=slice(offset,offset+chunk)
        x=means[sl]/sizes[ids[sl],2,None]*2
        condition=state['instances_embedding'][ids[sl]]
        t=x.new_full((len(x),1),frame/max(frames-1,1))
        xe,te=embed(x,cfg['x_multires']),embed(t,cfg['t_multires'])
        original=torch.cat([xe,te,condition],-1);h=original
        for i in range(cfg['D']):
            h=F.relu(F.linear(h,state[f'deform_network.linear.{i}.weight'],state[f'deform_network.linear.{i}.bias']))
            if i==cfg['D']//2:h=torch.cat([original,h],-1)
        output.append(F.linear(h,state['deform_network.gaussian_warp.weight'],state['deform_network.gaussian_warp.bias']))
        if cfg['deform_quat']:
            rotations.append(F.linear(h,state['deform_network.gaussian_rotation.weight'],state['deform_network.gaussian_rotation.bias']))
    return torch.cat(output),torch.cat(rotations) if rotations else None


def rigid_geometry(state,frame,delta=None,delta_q=None):
    ids=state['points_ids'].flatten().long()
    valid=state['instances_fv'][frame].bool()[ids]
    q=F.normalize(state['instances_quats'][frame],dim=-1)[ids]
    local=state['_means'] if delta is None else state['_means']+delta
    xyz=(quat_matrix(q)@local[...,None]).squeeze(-1)+state['instances_trans'][frame][ids]
    lq=state['_quats'] if delta_q is None else F.normalize(state['_quats'],dim=-1)+delta_q
    return xyz,F.normalize(quat_mul(q,F.normalize(lq,dim=-1)),dim=-1),valid


def skinning_transforms(rot,joints,parents):
    rel=joints.clone();rel[:,1:] -= joints[:,parents[1:]]
    bottom=rot.new_zeros(rot.shape[:-2]+(1,4));bottom[...,0,3]=1
    local=torch.cat([torch.cat([rot,rel[...,None]],-1),bottom],-2)
    chain=[local[:,0]]
    for i in range(1,len(parents)):chain.append(chain[int(parents[i])]@local[:,i])
    global_t=torch.stack(chain,1)
    j=torch.cat([joints,torch.zeros_like(joints[...,:1])],-1)[...,None]
    result=global_t.clone();result[...,3:4] -= global_t@j
    return result


def smpl_geometry(state,frame):
    visible=state['instances_fv'][frame].bool()
    total=len(visible)
    means=state['_means'].reshape(total,-1,3)
    quats=state['_quats'].reshape(total,-1,4)
    if not visible.any():return means.reshape(-1,3),F.normalize(quats.reshape(-1,4),dim=-1),visible.repeat_interleave(means.shape[1])
    ids=state['points_ids'].flatten().long()
    # The supplied SMPL implementation reshapes by instance; require contiguous
    # point ordering instead of silently assigning another person's geometry.
    expected=torch.arange(total,device=ids.device).repeat_interleave(means.shape[1])
    if not torch.equal(ids,expected):raise ValueError('SMPL point order is not instance-contiguous')
    theta=torch.cat([state['instances_quats'][frame],state['smpl_qauts'][frame]],1)
    a=skinning_transforms(quat_matrix(theta[visible]),state['template.J_canonical'][visible],
                          state['template._template_layer.parents'].long())
    a=a@state['template.A0_inv'][visible]
    prefix='template.voxel_deformer.'
    if prefix+'lbs_voxel_base' in state:
        grid=(means-state[prefix+'offset'])/state[prefix+'scale']
        grid=grid.clone();grid[...,-1] *= state[prefix+'ratio']
        weights=state[prefix+'lbs_voxel_base']+state[prefix+'voxel_w_correction']
        w=F.grid_sample(weights[visible],grid[visible,:,None,None,:],align_corners=True,
                        mode='bilinear',padding_mode='border').squeeze(3,4).permute(0,2,1)
    else:w=state['template.W'][visible]
    if w.shape[-1]!=a.shape[1]:raise ValueError('Unexpected additional SMPL skinning channels')
    blend=torch.einsum('bnj,bjrc->bnrc',w,a)
    r,t=blend[...,:3,:3],blend[...,:3,3]
    out=means.clone();out[visible]=(r@means[visible,...,None]).squeeze(-1)+t
    out+=state['instances_trans'][frame,:,None,:]
    oq=F.normalize(quats,dim=-1).clone()
    oq[visible]=quat_mul(matrix_quat(r),oq[visible])
    return out.reshape(-1,3),F.normalize(oq.reshape(-1,4),dim=-1),visible[ids]


def lighting(state,frame,frames,camera_index):
    # Match training code's denominator num_full_images-1, not num_timesteps-1.
    emb=state['appearance_embedding.weight'];n=len(emb)
    t=emb.new_tensor([[frame/max(1,n-1)]])
    scales=state['time_encoding.scales']
    xb=(t[...,None,:]*scales[:,None]).reshape(1,-1)
    time=torch.cat([t,torch.sin(torch.cat([xb,xb+.5*math.pi],-1))],-1)
    original=torch.cat([time,emb[camera_index:camera_index+1]],-1);h=original
    layers=sorted(int(k.split('.')[2]) for k in state if k.startswith('mlp.layers.') and k.endswith('.weight'))
    skip=max(1,len(layers)//2)
    for i in layers:
        if i==skip:h=torch.cat([h,original],-1)
        h=F.linear(h,state[f'mlp.layers.{i}.weight'],state[f'mlp.layers.{i}.bias'])
        if i!=layers[-1]:h=F.relu(h)
    return h.reshape(-1,3)


def pbr_color(state,camera_position,light):
    albedo=torch.sigmoid(state['_features_dc'])
    normal=F.normalize(state['_normals'],dim=-1)
    view=F.normalize(camera_position-state['_means'],dim=-1)
    diffuse=albedo*.1 if light is None else albedo*sh_eval(light,normal).clamp_min(0)
    amplitude=torch.sigmoid(state['_asg_amplitude'])
    direction=F.normalize(state['_asg_dir'],dim=-1)
    exponent=10/(torch.sigmoid(state['_roughness'])+1e-6)
    spec=torch.zeros_like(diffuse)
    for i in range(direction.shape[1]):
        halfway=F.normalize(view+direction[:,i],dim=-1)
        dot=(normal*halfway).sum(-1,keepdim=True).clamp(0,1)
        spec+=amplitude[:,i]*dot.pow(exponent)
    hdr=diffuse+spec
    return hdr/(1+hdr)
