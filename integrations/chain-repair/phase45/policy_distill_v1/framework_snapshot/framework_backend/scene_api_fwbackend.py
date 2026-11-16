"""scene_api substitute: SceneSession defaulting to FrameworkSceneBackend.

DIAGNOSTIC, UNCERTIFIED. Same full re-export pattern as the other phase26
shims (whitelist shims failed twice on lazy imports; never again). Only the
default backend factory changes: observations are produced by the framework
renderer (bridge/framework_backend) instead of the HUGSIM native renderer.
"""

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

import importlib.util
import sys
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent            # bridge/framework_backend
ROOT = next(p for p in HERE.parents if (p/'integrations/sr-hugsim/scene_api.py').is_file())
JOB = ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4'
for _p in (str(HERE), str(ROOT / 'integrations/sr-hugsim')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    'scene_api_reviewed', ROOT / 'integrations/sr-hugsim/scene_api.py')
_real = importlib.util.module_from_spec(_spec)
sys.modules['scene_api_reviewed'] = _real
_spec.loader.exec_module(_real)

for _name in dir(_real):
    if not _name.startswith('_') and _name != 'SceneSession':
        globals()[_name] = getattr(_real, _name)
del _name


class SceneSession(_real.SceneSession):
    def __init__(self, backend_factory=None):
        if backend_factory is None:
            from backend import FrameworkSceneBackend
            backend_factory = FrameworkSceneBackend
        super().__init__(backend_factory)
