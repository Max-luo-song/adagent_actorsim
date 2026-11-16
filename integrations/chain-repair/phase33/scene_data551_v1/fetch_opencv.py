"""Download the official pinned wheel in bounded verified ranges."""

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

import hashlib,json,urllib.request,time,subprocess,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent
name='opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
with urllib.request.urlopen('https://pypi.org/pypi/opencv-python-headless/4.10.0.84/json',timeout=60) as r:meta=_hugsim_expand(json.load(r))
item=next(f for f in meta['urls'] if f['filename']==name);expected=item['digests']['sha256'];total=item['size'];cache=HERE/'dependency_cache';cache.mkdir(exist_ok=True);target=cache/name
if not target.exists():
 with target.with_suffix('.part').open('wb') as f:
  for offset in range(0,total,4000000):
   end=min(offset+4000000,total)-1
   for attempt in range(4):
    try:
     req=urllib.request.Request(item['url'],headers={'Range':'bytes=%d-%d'%(offset,end),'Accept-Encoding':'identity'})
     with urllib.request.urlopen(req,timeout=90) as r:
      if r.status!=206 or not r.headers.get('Content-Range','').startswith('bytes %d-'%offset):raise ValueError('Server did not honor range')
      data=r.read(end-offset+2)
     if len(data)!=end-offset+1:raise ValueError('Incomplete range')
     break
    except Exception:
     if attempt==3:raise
   f.write(data);f.flush();print('OpenCV downloaded',end+1,'/',total,flush=True)
 target.with_suffix('.part').rename(target)
digest=hashlib.sha256(target.read_bytes()).hexdigest()
if digest!=expected:raise ValueError('Official wheel SHA256 mismatch: '+digest)
print('Official wheel SHA256 verified:',digest,flush=True)
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(HERE/'runtime_deps'),str(target)],check=True)
(HERE/'opencv_dependency.json').write_text(json.dumps({'package':'opencv-python-headless','version':'4.10.0.84','official_wheel_url':item['url'],'sha256':digest,'bytes':total},indent=2)+'\n')
