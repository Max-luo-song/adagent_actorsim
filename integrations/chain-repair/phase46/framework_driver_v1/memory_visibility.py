"""Whole-box ray exit evidence; occlusion/missing depth never means free.

Only a complete in-frame view can contradict an old nominal OBB. Every pixel
whose ray intersects the volume must have reliable depth beyond the far exit
plus the existing .5m error budget. This is a declared sensor model, not a
physical accuracy certificate. No asset truth or future frames enter.
"""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import json
pass  # Path is provided by the portable boundary above
import numpy as np
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name("memory_lifecycle_policy.json").read_text()))
from depth_contract import reliable_mask,geometry_depth


def box_free(box,obs,info,error_budget=None):
    error_budget=float(POLICY["depth_error_budget_m"] if error_budget is None else error_budget)
    if not np.isfinite(error_budget) or error_budget<=0:raise ValueError('Invalid depth error budget')
    if not obs or info.get('depth_encoding')!='expected_camera_z_v1':
        return dict(free=False,reason='missing_or_unsupported_depth')
    box=np.asarray(box,dtype=float)
    if box.shape!=(9,) or not np.isfinite(box).all() or np.any(box[3:6]<=0):
        raise ValueError('Invalid historical box')
    c,s=np.cos(box[6]),np.sin(box[6]);axes=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    center=box[:3]+[0.,0.,box[5]/2];half=box[3:6]/2
    corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])*half
    corners=corners@axes.T+center;views=[]
    for name in sorted(obs.get('depth',{})):
        params=info.get('cam_params',{}).get(name)
        if params is None:continue
        if name not in obs.get('depth_alpha',{}) and name not in obs.get('surface_valid',{}):
            views.append(dict(camera=name,reason='missing_reliability_evidence'));continue
        depth=np.asarray(geometry_depth(obs,name));valid=np.asarray(reliable_mask(obs,name))
        intr=params['intrinsic'];w,h=int(intr['W']),int(intr['H'])
        if depth.shape!=(h,w) or valid.shape!=depth.shape:raise ValueError('Depth/intrinsic shape mismatch')
        transform=np.asarray(params['l2c'],dtype=float);r=transform[:3,:3];t=transform[:3,3]
        if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(r@r.T,np.eye(3),atol=1e-5):raise ValueError('Invalid camera transform')
        cp=corners@r.T+t
        if np.min(cp[:,2])<=.1:views.append(dict(camera=name,reason='near_plane_or_behind'));continue
        fx=w/(2*np.tan(intr['fovx']/2));fy=h/(2*np.tan(intr['fovy']/2))
        uv=cp[:,:2]/cp[:,2,None]*[fx,fy]+[intr['cx'],intr['cy']]
        if (uv<0).any() or (uv[:,0]>w-1).any() or (uv[:,1]>h-1).any():
            views.append(dict(camera=name,reason='not_fully_in_frame'));continue
        x0,y0=np.floor(uv.min(0)).astype(int);x1,y1=np.ceil(uv.max(0)).astype(int)
        origin=(-r.T@t-center)@axes
        rays_checked=0;all_free=True
        # Chunks bound memory for very close/large historical boxes.
        for yy in range(y0,y1+1,16):
            y,x=np.mgrid[yy:min(yy+16,y1+1),x0:x1+1];x=x.reshape(-1);y=y.reshape(-1)
            direction=np.c_[(x-intr['cx'])/fx,(y-intr['cy'])/fy,np.ones(len(x))]@r@axes
            parallel=np.abs(direction)<1e-12
            lower=np.divide(-half-origin,direction,out=np.full_like(direction,-np.inf),where=~parallel)
            upper=np.divide(half-origin,direction,out=np.full_like(direction,np.inf),where=~parallel)
            enter=np.minimum(lower,upper).max(1);exit=np.maximum(lower,upper).min(1)
            hit=(exit>=np.maximum(enter,.1))&~np.any(parallel&(np.abs(origin)>half),axis=1)
            if not hit.any():continue
            x=x[hit];y=y[hit];far=exit[hit];z=depth[y,x]
            free=valid[y,x]&np.isfinite(z)&(z>0)&(z-error_budget>far)
            rays_checked+=len(z)
            if not free.all():all_free=False;break
        views.append(dict(camera=name,rays_checked=rays_checked,free=bool(all_free and rays_checked)))
        if all_free and rays_checked:
            return dict(free=True,camera=name,rays_checked=rays_checked,error_budget_m=error_budget,
                physical_error_bound_certified=False,scope='entire nominal historical OBB',views=views)
    return dict(free=False,reason='occluded_partial_unreliable_or_unseen',views=views)
