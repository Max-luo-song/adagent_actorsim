"""Bounded static RGB-D replay through the unified scene API; no driving/inference."""

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

import json,pickle
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_api import SceneSession,SceneSpec,BASE

def run(request_path,output):
 from build_source_sequence import training_lidar_mount
 from uniad_bridge import to_uniad
 from PIL import Image
 request=_hugsim_expand(json.loads(Path(request_path).read_text()))
 if set(request)!={'schema','scene_manifest','insert_request','views','reference_rgb_root'} or request['schema']!='static_depth_probe_v1' or not 1<=len(request['views'])<=8:raise ValueError('Invalid bounded static depth request')
 for field in ('scene_manifest','insert_request'):
  p=Path(request[field]).resolve()
  if BASE not in p.parents or not p.is_file():raise ValueError('Request inputs must be workspace files')
 spec=SceneSpec.read(request['scene_manifest']);insert=_hugsim_expand(json.loads(Path(request['insert_request']).read_text()));mount=training_lidar_mount(BASE/'code/HUGSIM/configs/sim/nuscenes_camera.yaml');output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
 with SceneSession() as scene:
  scene.load_scene(request['scene_manifest']);scene.edit(insert)
  for view in request['views']:
   if set(view)!={'step','pose','timestamp_s','speed_mps','steering_rad'} or not isinstance(view['step'],int) or view['step']<0:raise ValueError('Invalid static view')
   packet=scene.render(np.array(view['pose']),float(view['timestamp_s']));obs,info=to_uniad(packet,mount,speed_mps=view['speed_mps'],steering_rad=view['steering_rad'],command=2,scene_token='static-depth-probe')
   folder=output/f"step_{view['step']:03d}";folder.mkdir();differences={}
   with (folder/'observation.pkl').open('wb') as f:pickle.dump((obs,info),f)
   for name,image in obs['rgb'].items():
    Image.fromarray(image).save(folder/(name+'.png'));prior=np.asarray(Image.open(Path(request['reference_rgb_root'])/name/f"{view['step']:03d}.png"));differences[name]=int(abs(image.astype(int)-prior.astype(int)).max())
   rows.append(dict(step=view['step'],packet=str(folder/'observation.pkl'),rgb_max_errors=differences));(output/'run_report.json').write_text(json.dumps(dict(rows=rows,static_render_only=True,training=False,control_actions_executed=False),indent=2))
