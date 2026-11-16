"""Matched edge-focused ablations with supported, bounded background footprints."""
import numpy as np
import torch

PROFILES={
    'edge_color':dict(epochs=6,footprint=False,dc_lr=5e-4,rest_lr=1.25e-4,l1_weight=.8,ssim_weight=.2,edge_weight=.8),
    'footprint':dict(epochs=6,footprint=True,dc_lr=5e-4,rest_lr=1.25e-4,scale_lr=2e-3,opacity_lr=5e-3,
                     l1_weight=.8,ssim_weight=.2,edge_weight=.8),
}
COLOR_FIELDS={('Background','_features_dc'),('RoadNodes','_features_dc'),('RoadNodes','_features_rest')}
FOOT_FIELDS={('Background','_scales'),('Background','_opacities')}
SCALE_LOW=float(np.log(.85));SCALE_HIGH=float(np.log(1.05));OPACITY_DELTA=.2


def select_parameters(trainer,profile):
    trainer.requires_grad_(False)
    for m in trainer.models.values():m.requires_grad_(False)
    if not getattr(trainer.models['Background'],'use_pbr',False):raise ValueError('Expected the verified PBR background')
    selected={};groups=[]
    for model,field in sorted(COLOR_FIELDS | (FOOT_FIELDS if profile['footprint'] else set())):
        p=getattr(trainer.models[model],field)
        if not torch.isfinite(p).all():raise ValueError('Nonfinite source parameter '+model+'.'+field)
        p.requires_grad_(True);selected[model,field]=p
        lr=profile['scale_lr'] if field=='_scales' else profile['opacity_lr'] if field=='_opacities' else profile['rest_lr'] if field=='_features_rest' else profile['dc_lr']
        groups.append(dict(name=model+'.'+field,params=[p],lr=lr))
    return selected,groups


def bounds(key):
    if key==('Background','_scales'):return SCALE_LOW,SCALE_HIGH
    if key==('Background','_opacities'):return -OPACITY_DELTA,OPACITY_DELTA
    if key not in COLOR_FIELDS:raise ValueError('Forbidden parameter')
    radius=.05 if key[1]=='_features_rest' else .1
    return -radius,radius


def restrict_gradients(selected,support):
    for key,p in selected.items():
        if key in FOOT_FIELDS and p.grad is not None:p.grad[~support]=0


def clamp_updates(selected,initial,support):
    with torch.no_grad():
        for key,p in selected.items():
            lo,hi=bounds(key);p.copy_(torch.maximum(torch.minimum(p,initial[key]+hi),initial[key]+lo))
            if key in FOOT_FIELDS:p[~support]=initial[key][~support]


def footprint_anchor(selected,initial,support):
    sample=next(iter(selected.values()))
    result=sample.sum()*0
    if not bool(support.any()):return result
    for key in FOOT_FIELDS:
        if key in selected:
            radius=max(abs(x) for x in bounds(key))
            result=result+((selected[key][support]-initial[key][support])/radius).square().mean()
    return result


def depth_terms(depth,opacity,reference):
    depth=depth.squeeze(-1);opacity=opacity.squeeze(-1)
    valid=reference['mask'].bool()
    if depth.shape!=valid.shape or opacity.shape!=valid.shape:raise ValueError('Reference depth shape changed')
    if not bool(valid.any()):raise ValueError('Empty static depth reference')
    if not torch.isfinite(depth[valid]).all() or not torch.isfinite(opacity[valid]).all():raise ValueError('Nonfinite student depth/opacity')
    d=reference['depth'];a=reference['opacity']
    loss=(depth[valid].clamp_min(.1).log()-d[valid].clamp_min(.1).log()).abs().mean()
    alpha=(opacity[valid]-a[valid]).abs().mean()
    return loss,alpha


def depth_metrics(depth,opacity,reference):
    depth=depth.squeeze(-1);opacity=opacity.squeeze(-1);mask=reference['mask'].bool()
    error=((depth-reference['depth']).abs()/reference['depth'].clamp_min(.1))[mask]
    if not error.numel() or not torch.isfinite(error).all():raise ValueError('Invalid depth reference metrics')
    return dict(mean_relative_depth=float(error.mean()),p95_relative_depth=float(torch.quantile(error,.95)),
                mean_opacity_change=float((opacity-reference['opacity']).abs()[mask].mean()),
                new_hole_fraction=float((opacity[mask]<.5).float().mean()),valid_pixels=int(mask.sum()))


def visible_centers(means,camera_to_world,K,reference):
    """Conservative visibility proxy from original-view render depth, not LiDAR GT."""
    xyz=np.asarray(means,dtype=np.float64);pose=np.asarray(camera_to_world,dtype=np.float64);intr=np.asarray(K,dtype=np.float64)
    depth=np.asarray(reference['depth']);mask=np.asarray(reference['mask'],dtype=bool)
    h,w=depth.shape;p=(xyz-pose[:3,3])@pose[:3,:3];z=p[:,2]
    denominator=np.maximum(z,.01)
    u=np.rint(intr[0,0]*p[:,0]/denominator+intr[0,2]).astype(np.int64)
    v=np.rint(intr[1,1]*p[:,1]/denominator+intr[1,2]).astype(np.int64)
    valid=(z>.2)&(z<120)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
    indices=np.flatnonzero(valid);out=np.zeros(len(xyz),dtype=bool)
    yy=v[indices];xx=u[indices];surface=depth[yy,xx]
    good=mask[yy,xx]&(np.abs(z[indices]-surface)<(.25+.02*surface))
    out[indices[good]]=True
    return out


class SupportAccumulator:
    def __init__(self,count):
        self.count=np.zeros(count,dtype=np.int16);self.current=None;self.visible=np.zeros(count,dtype=bool)
    def add(self,frame,visible):
        if self.current is not None and frame<self.current:raise ValueError('Support observations must be chronological')
        if self.current is not None and frame!=self.current:
            self.count+=self.visible;self.visible[:]=False
        self.current=frame;self.visible|=visible
    def finish(self):
        if self.current is not None:self.count+=self.visible
        self.current=None;self.visible[:]=False
        return self.count>=2


def validate_checkpoint(before,after,allowed,support):
    """Exact frozen-state check plus support and footprint bounds on changed tensors."""
    if not set(allowed)<=(COLOR_FIELDS|FOOT_FIELDS):raise ValueError('Invalid checkpoint update scope')
    changed=[]
    def walk(a,b,path=()):
        if isinstance(a,torch.Tensor):
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Checkpoint tensor shape/dtype changed')
            if torch.equal(a,b):return
            key=tuple(path[1:]) if len(path)==3 and path[0]=='models' else None
            if key not in allowed or not torch.isfinite(b).all():raise ValueError('Frozen tensor changed: '+'.'.join(path))
            delta=b-a;lo,hi=bounds(key)
            if float(delta.min())<lo-1e-6 or float(delta.max())>hi+1e-6:raise ValueError('Parameter trust region exceeded')
            if key in FOOT_FIELDS and not torch.equal(a[~support],b[~support]):raise ValueError('Unsupported background point changed')
            changed.append('.'.join(path))
        elif isinstance(a,dict):
            ak=set(a)-({'optimizer'} if not path else set());bk=set(b)-({'optimizer'} if not path else set())
            if ak!=bk:raise ValueError('Checkpoint keys changed')
            for k in ak:walk(a[k],b[k],path+(str(k),))
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Checkpoint sequence changed')
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+(str(i),))
        elif isinstance(a,np.ndarray):
            if not np.array_equal(a,b):raise ValueError('Frozen array changed')
        elif a!=b:raise ValueError('Frozen metadata changed')
    walk(before,after)
    if not changed:raise ValueError('No model update')
    return sorted(changed)
