"""Bounded, isolated model-input experiments; never a production override."""
VARIANTS = ('reference', 'no_camera_embedding', 'fresh_frame', 'letterbox', 'letterbox_no_camera_embedding', 'letterbox_no_camera_embedding_pillar_down', 'letterbox_no_camera_embedding_pillar_up')


def requested_variants(sequence, preview):
    variants = sequence.get('diagnostic_variants')
    if variants is None:
        return None
    if not preview:
        raise ValueError('Model variants require quarantined perception preview')
    if (not isinstance(variants, list) or not variants or len(variants) > len(VARIANTS)
            or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants)):
        raise ValueError('Invalid bounded diagnostic variants')
    return variants


def configure_variant(core, variant):
    """Caller restores these switches after each independent replay."""
    if variant not in VARIANTS:
        raise ValueError('Unknown diagnostic variant')
    modules=list(core.modules())
    pillar_shift={'letterbox_no_camera_embedding_pillar_down':-.5,
                  'letterbox_no_camera_embedding_pillar_up':.5}.get(variant)
    encoders=[m for m in modules if hasattr(m,'point_sampling') and hasattr(m,'num_points_in_pillar') and hasattr(m,'pc_range')]
    if pillar_shift is not None and not encoders:raise ValueError('Pillar sampling hook not found')
    switches = [(module, bool(module.use_cams_embeds)) for module in modules
                if hasattr(module, 'use_cams_embeds')]
    if variant=='no_camera_embedding' or variant.startswith('letterbox_no_camera_embedding'):
        if not switches:
            raise ValueError('Camera embedding switch not found')
        for module, _ in switches:
            module.use_cams_embeds = False
    if pillar_shift is not None:
        from types import MethodType
        for encoder in encoders:
            original=encoder.point_sampling
            def sampling(self,reference_points,pc_range,img_metas,_original=original,_shift=pillar_shift):
                shifted=reference_points.clone()
                shifted[...,2]+=_shift/(pc_range[5]-pc_range[2])
                return _original(shifted,pc_range,img_metas)
            encoder.point_sampling=MethodType(sampling,encoder)
            switches.append((encoder,original,'point_sampling'))
    return switches


def restore_variant(switches):
    for item in reversed(switches):
        if len(item)==3:
            module,value,attribute=item;setattr(module,attribute,value)
        else:
            module,value=item;module.use_cams_embeds=value
