"""Color-only controlled experiments; scene geometry and actors stay immutable."""
import numpy as np
import torch
import torch.nn.functional as F

PROFILES = {
    # Same steps/order/loss as v1; isolate the DC learning-rate change.
    'lr_probe': dict(epochs=3, dc_lr=1e-3, rest_lr=1e-4, schedule='v1_blocks',
                     l1_weight=1., ssim_weight=0., edge_weight=0., trust_region=False),
    # Controlled optimization/scheduling comparison, without structural loss.
    'balanced': dict(epochs=12, dc_lr=1e-3, rest_lr=1.25e-4, schedule='interleaved',
                     l1_weight=1., ssim_weight=0., edge_weight=0., trust_region=True),
    # Identical to balanced except the loss terms.
    'structure': dict(epochs=12, dc_lr=1e-3, rest_lr=1.25e-4, schedule='interleaved',
                      l1_weight=.8, ssim_weight=.2, edge_weight=.1, trust_region=True),
}
ADAM_EPS = 1e-8
SEED = 20261108


def select_parameters(trainer, profile):
    trainer.requires_grad_(False)
    for model in trainer.models.values():model.requires_grad_(False)
    selected={};groups=[]
    for name in ['Background','RoadNodes']:
        model=trainer.models.get(name)
        if model is None:continue
        fields=['_features_dc']
        if not getattr(model,'use_pbr',False):fields.append('_features_rest')
        for field in fields:
            p=getattr(model,field,None)
            if p is None or not p.numel():continue
            p.requires_grad_(True);selected[(name,field)]=p
            groups.append(dict(name=name+'.'+field,params=[p],
                lr=profile['dc_lr'] if field=='_features_dc' else profile['rest_lr']))
    if not groups:raise ValueError('No active color parameters')
    return selected,groups


def epoch_schedule(groups, profile, rng):
    if not groups['novel'] or not groups['original']:raise ValueError('Novel and real GT replay required')
    if profile['schedule']=='v1_blocks':
        return [groups[k][i] for k in ['novel','original','original'] for i in rng.permutation(len(groups[k]))]
    if profile['schedule']!='interleaved':raise ValueError('Unknown schedule')
    order={k:rng.permutation(len(v)) for k,v in groups.items()}
    count=max(len(v) for v in groups.values())
    return [groups[k][order[k][i%len(groups[k])]] for i in range(count) for k in ['novel','original']]


def _ssim_error(pred,target,mask,window=11):
    if min(pred.shape[:2])<window:return pred.sum()*0,0
    # Valid windows only: excluded actor/sky pixels cannot enter SSIM statistics.
    valid=F.avg_pool2d(mask.float()[None,None],window,1)[0,0] >= 1.-1e-7
    count=int(valid.sum().detach())
    if not count:return pred.sum()*0,0
    x=pred.permute(2,0,1)[None];y=target.permute(2,0,1)[None]
    data=torch.cat([x,y,x*x,y*y,x*y],dim=1)
    axis=torch.arange(window,device=pred.device,dtype=pred.dtype)-(window-1)/2
    kernel=torch.exp(-axis.square()/(2*1.5**2));kernel=kernel/kernel.sum()
    horizontal=kernel.reshape(1,1,1,window).expand(15,1,1,window).contiguous()
    vertical=kernel.reshape(1,1,window,1).expand(15,1,window,1).contiguous()
    stats=F.conv2d(F.conv2d(data,horizontal,groups=15),vertical,groups=15)
    mx,my,xx,yy,xy=stats.split(3,dim=1)
    vx=(xx-mx*mx).clamp_min(0);vy=(yy-my*my).clamp_min(0)
    bound=(vx*vy+1e-12).sqrt();cov=(xy-mx*my).clamp(-bound,bound)
    score=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    error=(1-score.clamp(-1,1))/2
    return error[0,:,valid].mean(),count


def loss_terms(pred,target,mask):
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[-1]!=3 or mask.shape!=pred.shape[:2] or mask.dtype!=torch.bool:
        raise ValueError('RGB/mask shape or dtype mismatch')
    if not bool(mask.any()) or not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Empty mask or nonfinite RGB')
    l1=(pred-target).abs()[mask].mean()
    ssim,count=_ssim_error(pred,target,mask)
    mx=mask[:,1:]&mask[:,:-1];my=mask[1:]&mask[:-1]
    dx=(pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])
    dy=(pred[1:]-pred[:-1])-(target[1:]-target[:-1])
    edge_sum=dx.abs()[mx].sum()+dy.abs()[my].sum()
    edge_count=int(mx.sum().detach())+int(my.sum().detach())
    edge=edge_sum/(3*max(edge_count,1))
    return dict(l1=l1,ssim=ssim,edge=edge,ssim_windows=count,edge_pairs=edge_count)


def objective(terms,profile):
    return sum(profile[k+'_weight']*terms[k] for k in ['l1','ssim','edge'])


def clamp_color_update(selected,initial):
    with torch.no_grad():
        for key,p in selected.items():
            radius=.25 if key[1]=='_features_dc' else .1
            p.copy_(torch.maximum(torch.minimum(p,initial[key]+radius),initial[key]-radius))


def parameter_stats(selected,initial,optimizer):
    result={}
    for key,p in selected.items():
        stride=max(1,p.numel()//8192);delta=(p.detach().flatten()[::stride]-initial[key].flatten()[::stride]).abs()
        grad=p.grad
        state=optimizer.state[p]
        if grad is not None:
            sample=grad.detach().flatten()[::stride].abs()
            step=float(state['step']);beta2=optimizer.param_groups[0]['betas'][1]
            rms=state['exp_avg_sq'].flatten()[::stride].sqrt()/np.sqrt(1-beta2**step)
            attenuation=(rms/(rms+ADAM_EPS)).mean()
            result['.'.join(key)]=dict(grad_mean=float(sample.mean()),grad_max=float(sample.max()),
                sampled_delta_mean=float(delta.mean()),sampled_delta_max=float(delta.max()),
                sampled_at_trust_radius_fraction=float((delta >= (.25 if key[1]=='_features_dc' else .1)-1e-6).float().mean()),
                adam_rms_to_eps_factor=float(attenuation),gradient_present=True)
        else:result['.'.join(key)]=dict(gradient_present=False,sampled_delta_mean=float(delta.mean()))
    return result
