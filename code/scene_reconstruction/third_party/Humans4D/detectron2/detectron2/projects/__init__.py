
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

# Copyright (c) Facebook, Inc. and its affiliates.
import importlib.abc
import importlib.util
pass  # Path is provided by the portable boundary above

__all__ = []

_PROJECTS = {
    "point_rend": "PointRend",
    "deeplab": "DeepLab",
    "panoptic_deeplab": "Panoptic-DeepLab",
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "projects"

if _PROJECT_ROOT.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _D2ProjectsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("detectron2.projects."):
                return
            project_name = name.split(".")[-1]
            project_dir = _PROJECTS.get(project_name)
            if not project_dir:
                return
            target_file = _PROJECT_ROOT / f"{project_dir}/{project_name}/__init__.py"
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    import sys

    sys.meta_path.append(_D2ProjectsFinder())
