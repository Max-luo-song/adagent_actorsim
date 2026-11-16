
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

import re
pass  # Path is provided by the portable boundary above
from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
except ImportError as e:
    raise Exception(
        """
You must install PyTorch prior to installing DensePose:
pip install torch

For more information:
    https://pytorch.org/get-started/locally/
    """
    ) from e


def get_detectron2_current_version():
    """Version is not available for import through Python since it is
    above the top level of the package. Instead, we parse it from the
    file with a regex."""
    # Get version info from detectron2 __init__.py
    version_source = (Path(__file__).parents[2] / "detectron2" / "__init__.py").read_text()
    version_number = re.findall(r'__version__ = "([0-9\.]+)"', version_source)[0]
    return version_number


setup(
    name="detectron2-densepose",
    author="FAIR",
    version=get_detectron2_current_version(),
    url="https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "av>=8.0.3",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
        "opencv-python-headless>=4.5.3.56",
        "scipy>=1.5.4",
    ],
)
