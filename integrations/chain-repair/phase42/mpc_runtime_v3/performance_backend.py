"""Call-local performance features, with snapshot-local owners only."""

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

from contextvars import ContextVar
from contextlib import contextmanager
pass  # Path is provided by the portable boundary above
import json
NAMES=('rollout_prefix_reuse','batched_occupancy','bulk_pose_cache')
DEFAULT=_hugsim_expand(json.loads(Path(__file__).with_name('runtime_policy.json').read_text()))
_FLAGS=ContextVar('runtime_features',default={name:bool(DEFAULT.get(name,False)) for name in NAMES})
def enabled(name):return bool(_FLAGS.get().get(name,False))
@contextmanager
def use_features(policy):
 token=_FLAGS.set({name:bool(policy.get(name,False)) for name in NAMES})
 try:yield
 finally:_FLAGS.reset(token)
