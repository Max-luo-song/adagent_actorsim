"""Load the local reference-aware DiFix components, without remote code/network."""

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
import importlib
import sys


def components(config):
    weights = Path(config['weights'])
    # The ref checkpoint contains custom MV UNet and skip-connection VAE classes.
    # Loading stock diffusers classes would silently lose the reference behavior.
    sys.path[:0] = [config['src'], str(weights/'unet'), str(weights/'vae')]
    from pipeline_difix import DifixPipeline
    from unet_2d_condition import UNet2DConditionModel
    from autoencoder_kl import AutoencoderKL
    for module, expected in [('pipeline_difix',Path(config['src'])/'pipeline_difix.py'),
        ('unet_2d_condition',weights/'unet/unet_2d_condition.py'),
        ('autoencoder_kl',weights/'vae/autoencoder_kl.py')]:
        if Path(sys.modules[module].__file__).resolve() != expected.resolve():
            raise RuntimeError('DiFix component shadowed: '+module)
    return DifixPipeline, UNet2DConditionModel, AutoencoderKL


def load(config, device='cuda'):
    import torch
    pipeline, unet_class, vae_class = components(config)
    options = dict(local_files_only=True, torch_dtype=torch.float32 if device=='cpu' else torch.float16)
    unet = unet_class.from_pretrained(str(Path(config['weights'])/'unet'), **options)
    vae = vae_class.from_pretrained(str(Path(config['weights'])/'vae'), **options)
    pipe = pipeline.from_pretrained(config['weights'], unet=unet, vae=vae, **options).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
