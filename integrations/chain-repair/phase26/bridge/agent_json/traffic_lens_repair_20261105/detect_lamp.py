
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
from scipy import ndimage
pass  # Path is provided by the portable boundary above
from PIL import Image,ImageDraw
root=Path('outputs/waymo552_three_setup_20261018_043131/inputs')
out=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/agent_json/traffic_preparation_20261105')
frames=[100,110,112,120,130,140,160,180];canvas=Image.new('RGB',(4*420,2*350),(20,20,20));draw=ImageDraw.Draw(canvas);found={}
for i,f in enumerate(frames):
 im=Image.open(root/'images'/f'{f:03d}_0.jpg').convert('RGB');rgb=np.asarray(im);r,g,b=[rgb[:,:,j].astype(float) for j in range(3)]
 mask=((g>185)&(b>100)&(g>r*1.18)&(r<235)).astype(np.uint8);mask[:300]=0;mask[750:]=0;mask[:,:850]=0;mask[:,1400:]=0
 labels,n=ndimage.label(mask)
 stats=[];centers=[]
 for j,sl in enumerate(ndimage.find_objects(labels),start=1):
  if sl is None:continue
  ys,xs=np.nonzero(labels[sl]==j);ys+=sl[0].start;xs+=sl[1].start
  stats.append([int(xs.min()),int(ys.min()),int(np.ptp(xs)+1),int(np.ptp(ys)+1),len(xs)]);centers.append([xs.mean(),ys.mean()])
 pts=[];crop=im.crop((850,300,1400,750));di=ImageDraw.Draw(crop)
 for j in range(n):
  x,y,w,h,area=stats[j];cx,cy=centers[j]
  if area<5 or area>700 or w/h>2 or h/w>2:continue
  pts.append(dict(pixel=[float(cx),float(cy)],area=int(area),bbox=[int(x),int(y),int(w),int(h)]));di.rectangle([x-850-2,y-300-2,x-850+w+2,y-300+h+2],outline='red',width=2);di.text((x-850+5,y-300-12),str(len(pts)-1),fill='red')
 found[f]=pts;print(f,pts)
 crop.thumbnail((420,320));x=(i%4)*420;y=(i//4)*350;canvas.paste(crop,(x,y+25));draw.text((x+5,y+5),f'frame {f}',fill='white')
canvas.save(out/'green_blob_candidates.png');(out/'green_blob_candidates.json').write_text(json.dumps(found,indent=2))
