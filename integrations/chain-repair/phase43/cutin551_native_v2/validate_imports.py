"""Exercise actual launcher import order before loading any GPU model."""

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

import ast,sys
pass  # Path is provided by the portable boundary above
from paths import *
import run_cutin
run_cutin.configure()
import native_car_edit, native_scores, sensor_backend, scene_edit_session
for module,folder in [(native_car_edit,HERE),(native_scores,HERE),(sensor_backend,HERE/'framework_snapshot'),(scene_edit_session,HERE/'framework_snapshot/unified')]:
 if Path(module.__file__).resolve().parent!=folder:raise RuntimeError('Unexpected import origin: '+module.__file__)
backend,session=native_car_edit.make_types()
assert backend.__mro__[1] is sensor_backend.ClosedLoopFrameworkBackend
assert session.__mro__[1] is scene_edit_session.UnifiedSceneSession
for file in HERE.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
print('Actual launcher order binds to frozen Framework and native-car edit modules')
