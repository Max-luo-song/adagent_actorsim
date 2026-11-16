import torch

@torch.no_grad()
def relocate(env, instance, forward=None, right=0., edit_id=None, preview_only=False):
    state = env.models["RigidNodes"]
    if not 0 <= instance < state["instances_trans"].shape[1]:
        raise ValueError("车辆编号超出范围")
    if not bool(state["instances_fv"][0, instance]):
        raise ValueError("车辆首帧无效")

    if not hasattr(env, "_vehicle_originals"):
        env._vehicle_originals = {}
    originals = env._vehicle_originals
    if instance not in originals:
        originals[instance] = state["instances_trans"][:, instance].clone()
    original = originals[instance]
    delta = torch.zeros_like(original[0])
    if forward is not None:
        delta[0] = float(forward) - original[0, 0]
        delta[1] = -float(right) - original[0, 1]
    from edit_trust_adapter import require_edit_mutation, swept_instance_bounds
    # Include the entire animation, rather than certifying frame zero only.
    before = swept_instance_bounds(state, instance)
    size = float(torch.linalg.norm(state["instances_size"][instance]).item()/2)
    after_positions = (original+delta).detach().cpu().numpy()
    import numpy as np
    after = np.stack([after_positions.min(axis=0)-size, after_positions.max(axis=0)+size])
    require_edit_mutation(env, "move", dict(old_bounds=before, new_bounds=after), edit_id, preview_only)
    state["instances_trans"][:, instance].copy_(original + delta)
    env.cached_frame = None
    env.frame_cache = None
    return dict(instance=instance,
                original_source_xyz=original[0].tolist(),
                edited_source_xyz=(original[0] + delta).tolist(),
                translation_source_xyz=delta.tolist(),
                original_checkpoint_written=False)
