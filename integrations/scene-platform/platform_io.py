"""Standard-library-only platform paths, content identity and atomic records."""

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

import hashlib,json,os,re
pass  # Path is provided by the portable boundary above
ROOT=Path(__file__).resolve().parents[2]
CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
P26=CR/'phase26';LOCAL=Path(__file__).resolve().parent
DEFAULT_STORE=P26/'scene_platform_v1'
ID=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')

def read(path):
    def reject(s):raise ValueError('Nonfinite JSON: '+s)
    return _hugsim_expand(json.loads(Path(path).read_text(),parse_constant=reject))
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def asset(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def verify(value):
    if sha(value['path'])!=value['sha256']:raise ValueError('Frozen asset changed: '+value['path'])
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def identity(value):
    if not isinstance(value,str) or not ID.fullmatch(value):raise ValueError('Invalid identifier')
    return value

