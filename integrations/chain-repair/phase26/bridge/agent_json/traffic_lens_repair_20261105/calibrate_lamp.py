
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

import numpy as np,json
pass  # Path is provided by the portable boundary above
from scipy.optimize import least_squares
from scipy import ndimage
from PIL import Image,ImageDraw
root=Path('outputs/waymo552_three_setup_20261018_043131/inputs');out=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/agent_json/traffic_preparation_20261105')
cv=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]);intr=np.loadtxt(root/'intrinsics/0.txt');fx,fy,cx,cy,k1,k2,p1,p2,k3=intr;first=np.loadtxt(root/'ego_pose/000.txt');up=first[:3,:3].T@[0,0,1]
C={f:np.linalg.inv(first)@np.loadtxt(root/'ego_pose'/f'{f:03d}.txt')@np.loadtxt(root/'extrinsics/0.txt')@cv for f in range(198)}
def distortion(xy):
 x,y=xy[...,0],xy[...,1];r=x*x+y*y;rad=1+k1*r+k2*r*r+k3*r*r*r
 return np.stack([x*rad+2*p1*x*y+p2*(r+2*x*x),y*rad+p1*(r+2*y*y)+2*p2*x*y],axis=-1)
def project(point,f):
 q=np.linalg.inv(C[f])@np.r_[point,1];xy=distortion(q[:2]/q[2]);return xy*[fx,fy]+[cx,cy]
def ray(pixel,f):
 target=(np.asarray(pixel)-[cx,cy])/[fx,fy];xy=least_squares(lambda x:distortion(x)-target,target).x
 d=C[f][:3,:3]@np.r_[xy,1];return C[f][:3,3],d/np.linalg.norm(d)
found=_hugsim_expand(json.loads((out/'green_blob_candidates.json').read_text()));observations={}
for f in [100,110,112,120,130,140,160]:
 point=next(p for p in found[str(f)] if p['area']>70 and 450<p['pixel'][1]<550)
 observations[f]=point
train=[100,110,120,140];A=np.zeros((3,3));b=np.zeros(3)
for f in train:
 origin,d=ray(observations[f]['pixel'],f);w=np.eye(3)-np.outer(d,d);A+=w;b+=w@origin
initial=np.linalg.solve(A,b);fit=least_squares(lambda xyz:np.concatenate([project(xyz,f)-observations[f]['pixel'] for f in train]),initial)
point=fit.x;print('green_center',point.tolist(),'positive_depth',[float((np.linalg.inv(C[f])@np.r_[point,1])[2]) for f in train]);print('reproject_errors',{f:float(np.linalg.norm(project(point,f)-v['pixel'])) for f,v in observations.items()})
state_obs={}
canvas=Image.new('RGB',(6*220,280),(20,20,20));draw=ImageDraw.Draw(canvas)
for i,f in enumerate([176,180,185,190,195,197]):
 im=Image.open(root/'images'/f'{f:03d}_0.jpg').convert('RGB');rgb=np.asarray(im).astype(float);r,g,bb=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2];u,v=project(point,f);x0,x1=int(u-45),int(u+45);y0,y1=int(v-95),int(v+15)
 crop=im.crop((x0,y0,x1,y1)).resize((220,250),Image.Resampling.NEAREST);canvas.paste(crop,(220*i,30));draw.text((220*i+5,5),f'frame {f}',fill='white')
 for state,mask in [('red',(r>160)&(g<r*.65)&(bb<r*.65)),('yellow',(r>180)&(g>130)&(bb<g*.8)&(r>bb*1.5))]:
  roi=np.zeros(mask.shape,dtype=bool);roi[max(0,y0):min(y1,len(mask)),max(0,x0):min(x1,mask.shape[1])]=True
  labels,n=ndimage.label(mask&roi);candidates=[]
  for j in range(1,n+1):
   ys,xs=np.nonzero(labels==j)
   if len(xs)>=10:candidates.append(dict(pixel=[float(xs.mean()),float(ys.mean())],area=len(xs)))
  if candidates:
   selected=max(candidates,key=lambda x:x['area']);state_obs.setdefault(state,{})[f]=selected
 print('state',f,{s:d.get(f) for s,d in state_obs.items()})
canvas.save(out/'lamp_later_states.png')
profiles={'green':dict(center=point.tolist(),offset_up_m=0,observations=observations)}
for state,obs in state_obs.items():
 opt=least_squares(lambda height:np.concatenate([project(point+up*height[0],f)-entry['pixel'] for f,entry in obs.items()]),[.25 if state=='yellow' else .5])
 center=point+up*opt.x[0];profiles[state]=dict(center=center.tolist(),offset_up_m=float(opt.x[0]),observations=obs,reprojection_error_px={f:float(np.linalg.norm(project(center,f)-entry['pixel'])) for f,entry in obs.items()})
 print(state,'offset',opt.x,'errors',profiles[state]['reprojection_error_px'])
report=dict(green_training_frames=train,green_holdout_frames=[112,130,160],gravity_up_source=up.tolist(),lights=profiles,
 green_reprojection_error_px={f:float(np.linalg.norm(project(point,f)-v['pixel'])) for f,v in observations.items()},intrinsics_raw=intr.tolist())
(out/'lamp_multiframe_calibration.json').write_text(json.dumps(report,indent=2))
