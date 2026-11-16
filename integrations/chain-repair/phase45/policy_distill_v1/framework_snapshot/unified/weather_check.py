"""Resource structure gate, CPU-only. Does not claim weather inference."""

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

import json
pass  # Path is provided by the portable boundary above
root=Path(__file__).resolve().parents[2]/'framework_repo/WeatherEdit/background_editing/ckpts'
checks={str(root/'sd-turbo'/sub):(root/'sd-turbo'/sub).is_dir()
        for sub in ('tokenizer','text_encoder','vae','unet','scheduler')}
weight=root/'with_seg.pkl'
checks[str(weight)]=weight.is_file()
if weight.is_file():
    import torch
    data=torch.load(weight,map_location='cpu',weights_only=False)
    required=['rank_unet','l_target_modules_encoder','l_target_modules_decoder',
              'l_modules_others','rank_vae','vae_lora_target_modules','sd_vae_enc','sd_vae_dec']
    required += [f'sd_{weather}_{part}' for weather in ('rainy','snowy','foggy')
                 for part in ('encoder','decoder','other')]
    for key in required: checks[key]=key in data
print(json.dumps(dict(checks=checks,resource_structure_ready=all(checks.values()),
                     gpu_inference_verified=False,formal_evaluation_admitted=False),indent=2))
if not all(checks.values()): raise SystemExit(2)
