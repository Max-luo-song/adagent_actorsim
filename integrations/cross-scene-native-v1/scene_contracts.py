"""Scene-specific host bindings, with no policy or score changes."""

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

import hashlib,json,os,tempfile
pass  # Path is provided by the portable boundary above
ROOT=Path(__file__).resolve().parents[2]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
LOCAL=Path(__file__).resolve().parent
ASSETS=P26/'best_patch_generalization_v1/assets'
OUT=P26/'cross_scene_native_v1'
MAPS=OUT/'maps_principal_branch_v2'
DIAGNOSTICS=OUT/'diagnostics_principal_branch_v2'
NATIVE_ASSETS=OUT/'native_assets_principal_branch_v2'
REGISTRY=P26/'autonomous_patch_agent_v7/campaigns/patch_auto_v7/candidate_registry.json'
REPORTS={172:ROOT/'outputs/waymo172_three_setup_20261016_132050_295391/source_report.json',552:P26.parent/'phase4/dynamic552_frame80/source_report.json',621:ROOT/'outputs/avoidance_scene_search_20261022/scene621/source_report.json'}
REPORTS.update({s:P26.parent/'phase14/prepared'/str(s)/'source_report.json' for s in (31,233)})
COMMANDS={172:1,552:0,621:2,31:2,233:2} # Existing pre-registered navigation commands.
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def asset(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Changed artifact: '+a['path'])
def verify_pins(pins):
 for p,h in pins.items():verify({'path':p,'sha256':h})
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(name,p)
 finally:
  if os.path.exists(name):os.unlink(name)
def pins():return {str(p.resolve()):sha(p) for p in LOCAL.rglob('*') if p.suffix in ('.py','.sh') and '__pycache__' not in p.parts}
def policy(version):
 row=read(REGISTRY)[version];verify_pins(row['code_pins']);return row
