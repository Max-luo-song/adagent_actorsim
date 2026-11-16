"""Import the existing project APIs without loading GPU models or the driver."""

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
import sys
HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
PHASE=BRIDGE.parent
for parent in HERE.parents:
    if (parent/'integrations/sr-hugsim/scene_api.py').is_file():
        ROOT=parent;break
else:raise RuntimeError('project root not found')
for path in (BRIDGE/'unified',BRIDGE/'framework_backend',ROOT/'integrations/sr-hugsim',ROOT/'waymo23-native-import'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

# The framework constructor inserts F before importing utils.misc. Keep HUGSIM
# already present, at lower priority, so it cannot shadow framework utils.
HUG=ROOT/'code/HUGSIM'
if str(HUG) not in sys.path:sys.path.append(str(HUG))


def framework_import_check():
    from importlib.machinery import PathFinder
    framework=PHASE/'framework_repo'
    search=list(sys.path)
    for folder in (framework,HUG):
        if str(folder) not in search:search.insert(0,str(folder))
    result={}
    for name in ('utils','models'):
        spec=PathFinder.find_spec(name,search)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError('missing framework package: '+name)
        resolved=Path(next(iter(spec.submodule_search_locations))).resolve()
        if resolved!=(framework/name).resolve():
            raise RuntimeError('framework package shadowed: '+name+' -> '+str(resolved))
        result[name]=str(resolved)
    if not (framework/'utils/misc.py').is_file():raise RuntimeError('framework utils.misc missing')
    return result
