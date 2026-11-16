
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

import json,hashlib,os
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parent
OUT=LOCAL.parent
P26=OUT.parent
ROOT=P26.parents[5]
SCENES=(3,31,233,552)
PREVIOUS=P26/'two_pool_static_v1'
OMNI=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph'))
OLD_ASSETS=P26/'best_patch_generalization_v1/assets'
ASSETS=OUT/'assets'
MAPS=OUT/'maps'
REPORTS={s:P26.parent/'phase14/prepared'/str(s)/'source_report.json' for s in (3,23,31,233)}
REPORTS.update({172:ROOT/'outputs/waymo172_three_setup_20261016_132050_295391/source_report.json',552:P26.parent/'phase4/dynamic552_frame80/source_report.json',621:ROOT/'outputs/avoidance_scene_search_20261022/scene621/source_report.json'})
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Artifact changed: '+a['path'])
def write(p,data):
 p=Path(p).resolve()
 if not p.is_relative_to(OUT.resolve()):raise ValueError('Write outside isolated output')
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
 temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n');os.replace(temp,p)
def freeze(p,data):
 p=Path(p)
 if p.exists():
  if read(p)!=data:raise ValueError('Frozen artifact changed: '+str(p))
 else:write(p,data)
