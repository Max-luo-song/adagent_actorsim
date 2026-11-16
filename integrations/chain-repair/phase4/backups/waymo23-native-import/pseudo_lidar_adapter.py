"""Geometry-only post-detection adapter. No scene IDs, asset metadata or training."""
import numpy as np

POLICY = 'rendered_depth_center_temporal_gate_v2'


def refine_candidates(obs, info, boxes, scores, labels, detection_threshold=.4, history=None):
    from depth_surface_geometry import ground_plane, refine
    original = np.asarray(boxes, dtype=float).copy()
    corrected = original.copy()
    scores, labels = np.asarray(scores), np.asarray(labels)
    audit = dict(policy=POLICY, depth_source='rendered_expected_camera_z',
                 ground_truth_used=False, training=False, yaw_dimensions_velocity_modified=False,
                 original_boxes_preserved=True, candidates=[])
    if original.ndim != 2 or original.shape[1] < 7 or len(scores) != len(original) or len(labels) != len(original):
        raise ValueError('Unaligned detection arrays')
    depths = obs.get('depth', {})
    if not depths:
        audit['fallback_reason'] = 'missing_depth'
        return corrected, audit
    # Broken/missing calibration must never silently produce a correction.
    try:
        for name, depth in depths.items():
            depth = np.asarray(depth)
            camera = info['cam_params'][name]
            transform = np.asarray(camera['l2c'], dtype=float)
            if depth.ndim != 2 or transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('Invalid depth/calibration shape')
            if abs(np.linalg.det(transform[:3, :3])) < 1e-8:
                raise ValueError('Singular calibration')
        if info.get('depth_encoding')!='expected_camera_z_v1':
            audit['fallback_reason']='unknown_or_legacy_depth_encoding'
            return corrected,audit
        plane = ground_plane(obs, info)
        if plane is None:
            audit['fallback_reason'] = 'insufficient_observed_ground'
            return corrected, audit
        audit['ground_plane'] = plane.tolist()
    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
        audit['fallback_reason'] = 'invalid_depth_calibration: ' + str(exc)
        return corrected, audit
    for index, box in enumerate(original):
        # Same class, confidence, geometry and range rules in every scene.
        if (labels[index] != 0 or not np.isfinite(scores[index]) or scores[index] < detection_threshold
                or not np.isfinite(box).all() or np.any(box[3:6] <= 0)
                or np.linalg.norm(box[:2]) > 50):
            continue
        try:
            result = refine(obs, info, box, plane, fit_yaw=False)
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            result = dict(accepted=False, reason='invalid_surface: ' + str(exc), box=box.tolist())
        if history is not None:
            result=history.consider(result,box,info)
        proposed = np.asarray(result['box'], dtype=float)
        invariant = (proposed.shape == box.shape and np.isfinite(proposed).all()
                     and np.array_equal(proposed[2:], box[2:]))
        if result['accepted'] and invariant:
            corrected[index, :2] = proposed[:2]
        else:
            result['accepted'] = False
            if not invariant:
                result['reason'] = 'non_center_field_or_nonfinite_proposal'
        audit['candidates'].append(dict(candidate_index=index, original_box=box.tolist(),
            **{k: v for k, v in result.items() if k not in ('points', 'contour')}))
    audit['accepted_count'] = sum(row['accepted'] for row in audit['candidates'])
    return corrected, audit


def bind_frame(core, obs, info):
    """Replace the frame-local closure, so depth cannot leak across frames/scenes."""
    if not hasattr(core,'geometry_refinement_memory'):
        core.geometry_refinement_memory=TemporalRefinementGate()
    core.geometry_refinement_memory.begin(info,getattr(core,'_adapter_scene_token',info.get('scene_token')))
    core.geometry_candidate_refiner = lambda boxes, scores, labels, threshold: refine_candidates(
        obs, info, boxes, scores, labels, threshold,history=core.geometry_refinement_memory)


class TemporalRefinementGate:
    """Allow a small surface gain only when prior accepted geometry agrees.

    Keeps static-calibrated and uncalibrated centers from alternating solely
    because an absolute residual improvement barely missed 2 cm.
    """
    def __init__(self):
        self.anchors=[];self.scene=None;self.time=None
    def begin(self,info,scene):
        from runtime_configuration import configured_reliability
        self.config=configured_reliability();now=float(info.get('timestamp',0.))
        if self.time is not None and (scene!=self.scene or now<=self.time or now-self.time>self.config.max_gap_seconds):self.anchors=[]
        self.scene=scene;self.time=now
        self.anchors=[a for a in self.anchors if now-a['time']<=self.config.max_gap_seconds]
        self.previous=list(self.anchors)  # Never use another query from this frame as temporal evidence.
    def consider(self,result,box,info):
        result=dict(result);proposal=np.asarray(result.get('proposed_box',result['box']),dtype=float)
        rotation=np.asarray(info['explicit_l2g_r_mat']);translation=np.asarray(info['explicit_l2g_t']);world=rotation@proposal[:3]+translation
        if not result['accepted'] and 'before_surface_residual_m' in result:
            before=result['before_surface_residual_m'];after=result['after_surface_residual_m']
            quality=(before>.01 and after<.01 and after<before*.5
                     and np.linalg.norm(proposal[:2]-box[:2])<1.95
                     and np.array_equal(proposal[2:],box[2:]) and np.isfinite(proposal).all())
            distance=np.inf
            if quality:
                for anchor in self.previous:
                    ratio=proposal[3:6]/anchor['dimensions']
                    if not ((ratio>=.5).all() and (ratio<=2.).all()):continue
                    displacement=anchor['velocity'][:2]*(self.time-anchor['time'])
                    delta=world[:2]-anchor['world'][:2]
                    fraction=np.clip(float(delta@displacement)/max(float(displacement@displacement),1e-12),0.,1.)
                    distance=min(distance,float(np.linalg.norm(delta-fraction*displacement)))
            if quality and distance<=self.config.obstacle_margin_m:
                result.update(accepted=True,box=proposal.tolist(),reason='temporally consistent visible-surface gain',
                              temporal_geometry_support=True,temporal_anchor_distance_m=distance)
        if result['accepted'] and np.isfinite(proposal).all() and np.array_equal(proposal[2:],box[2:]):
            velocity=rotation@np.r_[box[7:9],0.] if len(box)>=9 else np.zeros(3)
            self.anchors.append(dict(world=world,velocity=velocity,dimensions=box[3:6].copy(),time=self.time))
        return result
