
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

import json,time,argparse
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from portable_segformer import PortableSegformer
from scene_data import write_json,sha
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();meta=_hugsim_expand(json.loads((a.bundle/'source_manifest.json').read_text()));source=meta['frames_data'][0]['camera_files'];start=time.monotonic();model=PortableSegformer('cpu');loaded=time.monotonic();mask=model.predict(source['0:image']['path']);finished=time.monotonic();path=a.bundle/'semantic_probe_000_0.png';Image.fromarray(mask).save(path)
def comparison(pred,path):
 original=np.asarray(Image.open(path))>0;union=np.sum(pred|original);return {'equal':bool(np.array_equal(pred,original)),'iou':float(np.sum(pred&original)/union) if union else 1.,'disagree_pixels':int(np.sum(pred!=original))}
result={'model':model.audit(),'model_load_seconds':loaded-start,'inference_seconds':finished-loaded,'mask':str(path),'sha256':sha(path),'road_comparison':comparison(mask==0,source['0:road']['path']),'sky_comparison':comparison(mask==10,source['0:sky']['path']),'classes':{str(x):int(np.sum(mask==x)) for x in np.unique(mask)},'new_closed_loop_executed':False};write_json(a.bundle/'semantic_probe.json',result);print(json.dumps(result,indent=2),flush=True)
