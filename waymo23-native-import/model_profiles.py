"""Versioned scene-independent UniAD input adapters; no score/vehicle overrides."""
from runtime_configuration import model_input_profile


def resolve_model_profile(sequence):
    forbidden = {'train_native_regression', 'regression_adapter_checkpoint', 'training_epochs'}
    if forbidden.intersection(sequence):
        raise ValueError('Zero-new-training protocol rejects training and learned adapter options')
    name=sequence.get('model_input_profile','legacy')
    if name not in ('legacy','calibration_geometry_v1'):
        raise ValueError('Unknown model input profile')
    if name!='legacy' and sequence.get('diagnostic_variants') is not None:
        raise ValueError('Choose a named model profile or diagnostic variants, not both')
    profile=dict(model_input_profile())
    profile.update(resize_mode='stretch' if name=='legacy' else 'letterbox',
                   camera_embedding_policy='pretrained' if name=='legacy' else 'geometry_only')
    return name,profile


def apply_model_profile(core,profile):
    policy=profile['camera_embedding_policy']
    if policy not in ('pretrained','geometry_only'):
        raise ValueError('Unknown camera embedding policy')
    modules=[module for module in core.modules() if hasattr(module,'use_cams_embeds')]
    if not modules:raise ValueError('Camera embedding switches not found')
    for module in modules:module.use_cams_embeds=policy=='pretrained'
    return len(modules)


def bind_sequence_model_profile(source, sequence):
    """Carry an explicit shared input profile through rendering into inference."""
    if 'model_input_profile' in source:
        name, _ = resolve_model_profile({'model_input_profile': source['model_input_profile']})
        sequence['model_input_profile'] = name
    return sequence
