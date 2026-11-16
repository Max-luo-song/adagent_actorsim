"""Resolve declared local geometry dependencies independently of cwd/tests."""

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
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'waymo23-native-import').is_dir() and (p/'code/UniAD_SIM').is_dir())
GEOMETRY=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad'
ADAPTER=ROOT/'waymo23-native-import'

def install():
    expected={'box_geometry':GEOMETRY/'box_geometry.py','input_geometry':GEOMETRY/'input_geometry.py','depth_contract':ADAPTER/'depth_contract.py'}
    for name,path in expected.items():
        if not path.is_file():raise ImportError('Declared geometry dependency missing: '+str(path))
        loaded=sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve()!=path.resolve():raise ImportError('Conflicting geometry module: '+name)
    for directory in (ADAPTER,GEOMETRY):
        if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    return {name:str(path) for name,path in expected.items()}
