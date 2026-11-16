"""Arc-length reference comparison consistent with spatial path tracking."""
import numpy as np

def path(points):
 points=np.asarray(points,dtype=float)
 if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Invalid reference geometry')
 keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10];points=points[keep];length=np.linalg.norm(np.diff(points,axis=0),axis=1);return points,np.r_[0.,np.cumsum(length)]
def at(points,arc,s):
 s=np.asarray(s);out=np.stack([np.interp(s,arc,points[:,j]) for j in range(2)],axis=-1)
 if len(points)>1:
  extra=np.maximum(s-arc[-1],0.);out+=extra[...,None]*(points[-1]-points[-2])/(arc[-1]-arc[-2])
 return out

def moving_path_error(anchor_reference,anchor_rotation,anchor_translation,current_reference,rotation,translation):
 old=np.asarray(anchor_reference,dtype=float);new=np.asarray(current_reference,dtype=float)
 if old.shape!=(6,2) or new.shape!=(6,2):raise ValueError('Expected six-point neural paths')
 old3=np.c_[np.vstack([np.zeros(2),old]),np.zeros(7)];old_local=(old3@np.asarray(anchor_rotation).T+np.asarray(anchor_translation)-np.asarray(translation))@np.asarray(rotation)
 a,sa=path(old_local[:,:2]);b,sb=path(np.vstack([np.zeros(2),new]))
 if len(a)<2 or len(b)<2:return float('inf')
 segments=np.diff(a,axis=0);length=np.diff(sa);u=np.clip(np.sum(-a[:-1]*segments,axis=1)/length**2,0.,1.);u[-1]=max(0.,float(-a[-2]@segments[-1]/length[-1]**2));closest=a[:-1]+u[:,None]*segments;index=int(np.argmin(np.sum(closest**2,axis=1)));start=sa[index]+u[index]*length[index];limit=sb[-1]
 samples=np.unique(np.r_[0.,limit,sb,sa[(sa>=start)&(sa<=start+limit)]-start]);return float(np.max(np.linalg.norm(at(a,sa,start+samples)-at(b,sb,samples),axis=1)))
