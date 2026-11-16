"""Explicit camera-Z depth encoding conversion at the simulator boundary."""
import numpy as np
ACCUMULATED='accumulated_camera_z_v1'
EXPECTED='expected_camera_z_v1'
def expected_depth(depth,alpha,encoding):
    depth=np.asarray(depth,dtype=np.float32);alpha=np.asarray(alpha,dtype=np.float32)
    if depth.shape!=alpha.shape or not np.isfinite(depth).all() or not np.isfinite(alpha).all() or np.any(depth<0) or np.any(alpha<0) or np.any(alpha>1.00001):raise ValueError('Invalid depth/opacity arrays')
    if encoding==EXPECTED:return depth.copy()
    if encoding!=ACCUMULATED:raise ValueError('Unknown depth encoding; do not guess from render-mode name')
    result=np.zeros_like(depth);np.divide(depth,alpha,out=result,where=alpha>1e-8)
    return result

def depth_variance(second_moment,alpha,mean_depth):
    second=np.asarray(second_moment,dtype=np.float32);alpha=np.asarray(alpha,dtype=np.float32);mean=np.asarray(mean_depth,dtype=np.float32)
    if second.shape!=mean.shape or alpha.shape!=mean.shape or not np.isfinite(second).all():raise ValueError('Invalid depth moment')
    expected_second=np.zeros_like(second);np.divide(second,alpha,out=expected_second,where=alpha>1e-8)
    return np.maximum(expected_second-mean*mean,0.)

def _surface(obs,name):
    depth=obs.get('surface_depth',{}).get(name)
    valid=obs.get('surface_valid',{}).get(name)
    if (depth is None)!=(valid is None):raise ValueError('Surface depth and validity must be paired')
    if depth is None:return None,None
    depth=np.asarray(depth,dtype=np.float32);valid=np.asarray(valid)
    shape=np.asarray(obs['depth'][name]).shape
    if depth.shape!=shape or valid.shape!=shape or valid.dtype!=np.bool_:raise ValueError('Invalid surface observation shape/type')
    if np.any(valid&(~np.isfinite(depth)|(depth<=0))):raise ValueError('Invalid trusted surface depth')
    return depth,valid

def reliable_mask(obs,name):
    depth,valid=_surface(obs,name)
    if depth is not None:return valid.copy()
    values=np.asarray(obs['depth'][name]);valid=np.isfinite(values)&(values>0)
    if name in obs.get('depth_alpha',{}):valid &= obs['depth_alpha'][name]>=.9
    if name in obs.get('depth_variance',{}):valid &= obs['depth_variance'][name]<=.5**2
    return valid

def geometry_depth(obs,name):
    depth,_=_surface(obs,name)
    return depth if depth is not None else obs['depth'][name]
