
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
import hashlib,json
P=Path(__file__).resolve().parent
files=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['snapshot_manifest.json','launch_status.json'] and p.suffix in ['.py','.json','.cpp','.c','.h','.hpp','.cu','.so','.sh','.md'] and not any(name in p.parts for name in ['runs','checks'])]
pins={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(P/'snapshot_manifest.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
print('Frozen full-scene snapshot:',len(pins),'files')
