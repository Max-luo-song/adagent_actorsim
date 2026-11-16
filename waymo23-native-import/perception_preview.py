"""Quarantined image-only diagnostics, with no exported planning or controls."""
from scene_reliability import ReliabilityConfig


def preview_requested(sequence):
    requested = sequence.get('perception_preview_requested', False)
    if requested is not True and requested is not False:
        raise ValueError('Preview opt-in must be a boolean')
    if requested and sequence.get('diagnostic_preview_only') is not True:
        raise ValueError('Perception preview requires a diagnostic-only sequence')
    return requested


class QuarantinedPerceptionObserver:
    """Do not run the decision guard or make trust claims on unverified edits."""
    def __init__(self):
        self.config = ReliabilityConfig()

    def reset(self):
        pass

    def update(self, boxes, scores, labels, *args, **kwargs):
        return dict(diagnostic_preview_only=True, evaluation_admitted=False,
                    edit_trust_verified=False, raw_scores_modified=False,
                    candidate_count=len(scores), control_use_forbidden=True,
                    planning_outputs_exported=False, decision_guard_run=False)
