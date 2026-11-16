
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

import sys,json,numpy as np
pass  # Path is provided by the portable boundary above
p=Path('outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/bridge/agent_json').resolve();sys.path.insert(0,str(p));import agent_bootstrap
from agent_catalog import Catalog
from scene_edit_session import saf
from ego_trajectory import compile_trajectory
from ego_attitude import horizon_metrics
spec=saf.SceneSpec.read(Catalog().scene('552'))
job=_hugsim_expand(json.loads((p/'examples/ego_horizon_comparison.json').read_text()))
old=compile_trajectory(job['steps'][1]['request']['args']['trajectory'],spec);new=compile_trajectory(job['steps'][2]['request']['args']['trajectory'],spec)
saved=_hugsim_expand(json.loads((p.parent/'unified/agent_json_runs/20261104T094154Z/session/trajectories/preview_path/packets.json').read_text()))
results=[]
for index in [0,20,40]:
 samples=[]
 for compiled in [old,new]:
  pose=np.asarray(compiled['samples'][index]['ego_to_world']);cameras={}
  for name,camera in saved['frames'][index]['new']['cameras'].items():
   cameras[name]=dict(K=camera['K'],camera_to_world=(pose@np.asarray(camera['camera_to_ego'])).tolist())
  samples.append(horizon_metrics(dict(ego_to_world=pose.tolist(),cameras=cameras),compiled['gravity_up_source']))
 results.append(dict(frame=80+index,before=samples[0],after=samples[1]))
print(json.dumps(dict(gravity_up_source=old['gravity_up_source'],results=results),indent=2))
