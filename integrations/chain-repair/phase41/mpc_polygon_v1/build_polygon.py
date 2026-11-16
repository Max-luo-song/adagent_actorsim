"""Build only this package's local polygon kernel; do not modify shared libraries."""

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

pass  # Path is provided by the portable boundary above
import hashlib,json,platform,subprocess
P=Path(__file__).resolve().parent;source=P/'polygon_kernel.cpp';folder=P/'polygon_build';folder.mkdir(exist_ok=True);library=folder/'libpolygon.so'
flags=['-O3','-std=c++17','-shared','-fPIC','-fno-fast-math','-ffp-contract=off']
subprocess.run(['g++',*flags,str(source),'-o',str(library)],check=True)
record=dict(abi=1,machine=platform.machine(),library='polygon_build/libpolygon.so',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),flags=flags)
(P/'polygon_build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
