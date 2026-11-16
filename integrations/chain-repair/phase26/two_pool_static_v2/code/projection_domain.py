"""Reject disconnected/folded branches of the calibrated lens projection."""
from functools import lru_cache
import numpy as np
REVISION='brown_conrady_principal_branch_v2'
POINT_FOLDER='point_semantics_principal_branch_v2'
@lru_cache(maxsize=64)
def principal_radius_squared(k1,k2,k3):
 limits=[]
 for coefficients in ((k3,k2,k1,1.),(7*k3,5*k2,3*k1,1.)):
  for root in np.roots(np.trim_zeros(coefficients,'f')):
   if abs(root.imag)<1e-10 and root.real>0:limits.append(float(root.real))
 return min(limits) if limits else np.inf

def valid_ray(pc,intrinsics):
 p=np.asarray(pc,float);k=np.asarray(intrinsics,float)
 if p.ndim!=2 or p.shape[1]!=3 or k.shape!=(9,):raise ValueError('Invalid calibrated ray input')
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  x=p[:,0]/p[:,2];y=p[:,1]/p[:,2];r2=x*x+y*y;k1,k2,p1,p2,k3=k[4:];g=1+k1*r2+k2*r2*r2+k3*r2*r2*r2;dg=k1+2*k2*r2+3*k3*r2*r2
  derivative=1+3*k1*r2+5*k2*r2*r2+7*k3*r2*r2*r2
  xx=g+2*x*x*dg+2*p1*y+6*p2*x;yy=g+2*y*y*dg+6*p1*y+2*p2*x;xy=2*x*y*dg+2*p1*x+2*p2*y
  return np.isfinite(p).all(1)&(p[:,2]>.1)&(r2<principal_radius_squared(float(k1),float(k2),float(k3)))&(g>0)&(derivative>0)&(xx>0)&(yy>0)&(xx*yy-xy*xy>1e-12)

def wrap_point_projector(original,cv_to_ego_rotation):
 def project(points,extrinsic,intrinsic,wh):
  pixels,depth,valid=original(points,extrinsic,intrinsic,wh);basis=np.eye(4);basis[:3,:3]=cv_to_ego_rotation;T=np.asarray(extrinsic)@basis;inv=np.linalg.inv(T);pc=np.asarray(points)@inv[:3,:3].T+inv[:3,3]
  return pixels,depth,valid&valid_ray(pc,intrinsic)
 return project
