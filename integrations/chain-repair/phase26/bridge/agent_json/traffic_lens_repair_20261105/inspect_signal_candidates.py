
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

pass  # Path is provided by the portable boundary above
import numpy as np,torch,json
from PIL import Image,ImageDraw
p=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26');out=p/'bridge/agent_json/traffic_preparation_20261105';root=Path('outputs/waymo552_three_setup_20261018_043131/inputs')
m=torch.load(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'),map_location='cpu',mmap=True,weights_only=False)['models']['TrafficLightNodes']
origin=np.loadtxt(root/'ego_pose/000.txt');cv=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
canvas=Image.new('RGB',(4*320,3*360),(20,20,20));draw=ImageDraw.Draw(canvas);col=0
for index,frame in [(6,110),(9,112),(10,160),(11,169)]:
 pose=np.linalg.inv(origin)@np.loadtxt(root/'ego_pose'/f'{frame:03d}.txt');mask=m['points_ids'].flatten()==index
 rgb=(m['_features_dc'][mask]*.28209479+.5).clamp(0,1);alpha=m['_opacities'][mask].sigmoid().flatten();chrom=(rgb.amax(-1)-rgb.amin(-1)>.2)&(rgb.amax(-1)>.5)
 print('index',index,'chromatic alpha',alpha[chrom].tolist(),'DCrgb',rgb[chrom].tolist())
 center=m['instances_trans'][frame,index].numpy()
 for row,cid in enumerate([0,1,2]):
  original=Image.open(root/'images'/f'{frame:03d}_{cid}.jpg').convert('RGB');intr=np.loadtxt(root/'intrinsics'/f'{cid}.txt');k=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]])
  c2w=pose@np.loadtxt(root/'extrinsics'/f'{cid}.txt')@cv;local=np.linalg.inv(c2w)@np.r_[center,1];pixel=k@local[:3];u,v=pixel[:2]/pixel[2];x,y=int(round(u)),int(round(v));w,h=original.size
  visible=local[2]>0 and 0<=x<w and 0<=y<h
  draw.text((col*320+5,row*360+5),f'index={index} frame={frame} cam={cid} inframe={visible}',fill='white')
  if visible:
   crop=original.crop((x-80,y-80,x+80,y+80)).resize((320,320),Image.Resampling.NEAREST);canvas.paste(crop,(col*320,row*360+30))
  print('project',index,frame,cid,(round(u,1),round(v,1)),round(float(local[2]),1),visible)
 col+=1
canvas.save(out/'other_signal_source_crops.png')
