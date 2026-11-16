"""Matched-pose before/after renders, no neural inference or control."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import json,numpy as np
from PIL import Image,ImageDraw

def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 arm=h._load_arm_request(HERE/'arm.json');setup=h._setup(arm,out);frames=(0,5,10);saved={};rows=[]
 with h.SceneSession() as scene:
  scene.load_scene(out/'scene_manifest.json')
  for f in frames:
   packet=scene.render(setup['relative'][f],f/setup['spec'].fps)
   saved[f]={k:packet['cameras'][k]['rgb'].copy() for k in h.THREE_CAMERAS}
  scene.edit(_hugsim_expand(json.loads((HERE/'insert_request.json').read_text())))
  for f in frames:
   packet=scene.render(setup['relative'][f],f/setup['spec'].fps);folder=out/f'frame_{f:03d}';folder.mkdir();diffs={}
   sheet=Image.new('RGB',(1920,800),'#111111');draw=ImageDraw.Draw(sheet)
   for col,name in enumerate(h.THREE_CAMERAS):
    before=saved[f][name];after=packet['cameras'][name]['rgb'];Image.fromarray(before).save(folder/f'{name}_before.png');Image.fromarray(after).save(folder/f'{name}_after.png');diffs[name]=int(np.any(before!=after,axis=-1).sum())
    for row,(kind,pixels) in enumerate((('before',before),('after',after))):
     im=Image.fromarray(pixels);im.thumbnail((640,360));sheet.paste(im,(col*640,row*400+32));draw.text((col*640+8,row*400+8),f'Scene551 | original pose frame{f} | {name} | {kind}',fill='white')
   sheet.save(folder/'comparison.jpg',quality=93);rows.append(dict(source_frame=f,changed_pixels=diffs));print(json.dumps(rows[-1]),flush=True)
 if not all(r['changed_pixels']['CAM_FRONT']>=100 for r in rows):raise RuntimeError('Inserted car not visibly changing FRONT; no closed-loop launched')
 result=dict(passed=True,scope='Matched recorded-pose edit visibility only; no detection/ground contact/corridor/safety certification',rows=rows,control_executed=False,neural_inference_executed=False,formal_evaluation_admitted=False)
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
