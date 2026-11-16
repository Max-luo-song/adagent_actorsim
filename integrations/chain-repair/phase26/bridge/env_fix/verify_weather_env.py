"""Import/API check only. Does not initialize large models or require CUDA."""

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

import importlib.metadata as md
import json
import sys
pass  # Path is provided by the portable boundary above

here=Path(__file__).resolve().parent
required={}
for line in (here/'weather_requirements.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name,version=line.split('==');required[name]=version
for name,version in required.items():
    if md.version(name)!=version: raise RuntimeError(f'{name}: expected {version}, got {md.version(name)}')
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig
from diffusers.utils import USE_PEFT_BACKEND
assert USE_PEFT_BACKEND
for cls in (AutoencoderKL,UNet2DConditionModel):
    for method in ('from_pretrained','add_adapter','set_adapter'):
        assert callable(getattr(cls,method,None)),(cls.__name__,method)
# Import the actual WeatherEdit module, including cross_att; no model loading.
sys.path.insert(0,str(here.parents[1]/'framework_repo/WeatherEdit/background_editing/src'))
from cyclegan_turbo import CycleGAN_Turbo
import torch
print(json.dumps(dict(packages={n:md.version(n) for n in required},
    torch_version=torch.__version__,torch_module=torch.__file__,
    cuda_available=torch.cuda.is_available(),weather_source_imported=True,
    gpu_inference_verified=False,formal_evaluation_admitted=False),indent=2))
