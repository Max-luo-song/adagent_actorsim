
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

def validate(root):
 p=Path(root)
 from change_contract import validate as validate_change
 validate_change(p)
 d=_hugsim_expand(json.loads((p/'corridor_acceptance.json').read_text()))
 if not d.get('passed') or d.get('schema')!='forward_corridor_acceptance_v4' or d.get('gpu_executed') is not False:raise ValueError('Missing own corridor acceptance')
 if len(d.get('selected_checks',[]))<3 or not all(r['passed'] for r in d['selected_checks']):raise ValueError('Invalid selected plan proof')
 if len(d.get('left_checks',[]))<2 or not all(r['passed'] and r['side_corridor_verified'] for r in d['left_checks']):raise ValueError('No independently verified left corridor')
 for name,expected in d['artifact_sha256'].items():
  if hashlib.sha256((p/name).read_bytes()).hexdigest()!=expected:raise ValueError('Corridor proof changed: '+name)
 for row in d['replay_core_hashes']:
  for name,expected in row.items():
   if hashlib.sha256((p/name).read_bytes()).hexdigest()!=expected:raise ValueError('Replay implementation changed: '+name)
 return True
