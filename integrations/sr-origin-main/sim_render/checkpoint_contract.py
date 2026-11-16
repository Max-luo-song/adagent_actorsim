"""Feature checks for frozen-scene inference, before dataset/CUDA allocation."""


def require_supported_scene_features(model_config):
    pbr = [name for name, cfg in model_config.items()
           if cfg.get('ctrl', {}).get('use_pbr', False)]
    if pbr and 'GlobalLighting' not in model_config:
        raise ValueError('Frozen PBR requires the checkpoint GlobalLighting node; do not discard it')
    if any(name != 'Background' for name in pbr):
        raise ValueError('Legacy PBR compatibility currently supports static Background only')
    if pbr and model_config['Background'].get('type') != 'models.gaussians.VanillaGaussians':
        raise ValueError('Legacy PBR requires the material-aware VanillaGaussians implementation')
    lighting = model_config.get('GlobalLighting')
    if lighting is not None and lighting.get('type') != 'models.modules.GlobalLightingModule':
        raise ValueError('Unsupported GlobalLighting implementation')
