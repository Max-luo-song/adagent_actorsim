
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

import hashlib,json,platform,subprocess
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent
FLAGS=['-O3','-std=c++17','-shared','-fPIC','-fno-fast-math','-ffp-contract=off']
def build():
 source=HERE/'kernels.cpp';digest=hashlib.sha256(source.read_bytes()+json.dumps(FLAGS).encode()).hexdigest()
 folder=HERE/'build';folder.mkdir(exist_ok=True);library=folder/('kernels_'+digest[:16]+'.so')
 if not library.exists():subprocess.run(['g++',*FLAGS,str(source),'-o',str(library)],check=True)
 record=dict(abi=1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),flags=FLAGS,library=str(library),machine=platform.machine(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0])
 (HERE/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':build()
