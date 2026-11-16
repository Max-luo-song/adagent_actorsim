"""Frozen isolated Framework driving bundle; use unified framework launcher."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import json,hashlib

def verify():
 pins=_hugsim_expand(json.loads((HERE/'manifest.json').read_text()))
 for name,digest in pins.items():
  path=ROOT/name
  if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Pinned file changed: '+str(path))
 print('Pinned SHA256 verified:',len(pins),'files',flush=True)
