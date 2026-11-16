
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

import hashlib,json,os,time
pass  # Path is provided by the portable boundary above
ROOT=Path(__file__).resolve().parents[2]
CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
OUT=CR/'phase26/best_patch_generalization_v1'
SOURCE=CR/'phase33/scene_data551_v1'
DATA=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph'))
RAW_BASE=DATA/'data/waymo/processed/training'
REPORTS={172:ROOT/'outputs/waymo172_three_setup_20261016_132050_295391/source_report.json',552:CR/'phase4/dynamic552_frame80/source_report.json',621:ROOT/'outputs/avoidance_scene_search_20261022/scene621/source_report.json'}
COMMANDS={172:1,552:0,621:2}
POLICIES={'v13':ROOT/'integrations/light-reference-join-v13-r2','v15':ROOT/'integrations/light-tracking-curvature-v15-r2','v17':ROOT/'integrations/light-execution-horizon-v17-r2'}
def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Changed frozen asset: '+a['path'])
def context():
 scene=int(os.environ['GENERALIZATION_SCENE'])
 if scene not in REPORTS:raise ValueError('Unregistered scene')
 raw=RAW_BASE/str(scene);report=read(REPORTS[scene]);frames=len(list((raw/'ego_pose').glob('*.txt')))
 if frames!=report['frames']:raise ValueError('Source/report coverage differs')
 return scene,raw,Path(report['checkpoint']),frames
