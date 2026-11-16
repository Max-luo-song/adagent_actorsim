"""Failure mining is separate from patch acceptance and infrastructure health."""
import math

def score(summary):
    value = ((summary.get('native_scores') or {}).get('scores') or {}).get('executed_path', {}).get('hdscore')
    if isinstance(value, dict):
        lo, hi = value.get('lower'), value.get('upper')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi)) or abs(lo-hi)>1e-9:
            return None
        return float(lo)
    return None


def classify(original, edited, edit_valid, prefix_verified, minimum_score_drop=.01):
    result = dict(eligible=False, classification='unresolved', reasons=[], patch_started=False)
    if not edit_valid or not prefix_verified:
        return dict(result, classification='invalid_intervention', reasons=['edit audit or matched prefix missing'])
    if original.get('status') != 'completed' or original.get('any_anomaly') or original.get('stop_recovery_unresolved'):
        return dict(result, classification='unhealthy_original', reasons=['original scene must complete without unresolved anomaly'])
    if edited.get('status') in ('failed', 'initializing', 'resource_limited') or edited.get('error_stage'):
        return dict(result, classification='infrastructure_failure', reasons=[edited.get('termination_reason') or 'runtime failure'])
    reasons=[]
    anomalies=edited.get('anomalies', {})
    if anomalies.get('environment_collision'):
        if edited.get('collision_depends_only_on_unknown_semantics') is True:
            return dict(result, classification='uncertain_geometry', reasons=['collision depends on unknown semantics'])
        if edited.get('collision_depends_only_on_unknown_semantics') is not False:
            return dict(result, classification='uncertain_geometry', reasons=['collision provenance missing'])
        reasons.append('observed_collision')
    if anomalies.get('native_route_deviation') or anomalies.get('known_executed_native_dac_below_09'):
        reasons.append('route_or_native_drivable_area_failure')
    if edited.get('termination_reason')=='no_jointly_verified_action':
        reasons.append('no_jointly_verified_action')
    if anomalies.get('tracker_error_over_1m'):
        reasons.append('execution_tracking_failure')
    if anomalies.get('abnormal_stall'):
        reasons.append('commanded_motion_but_actual_stall')
    a,b=score(original),score(edited)
    # A lower score after yielding is not itself evidence of a controller bug.
    drop=a-b if a is not None and b is not None else None
    if not reasons:
        return dict(result, classification='score_tradeoff_or_no_failure' if drop is not None and drop>=minimum_score_drop else 'no_confirmed_failure',
                    reasons=[], original_hdscore=a, edited_hdscore=b, score_drop=drop)
    return dict(result, eligible=True, classification='control_failure_candidate', reasons=reasons,
                original_hdscore=a, edited_hdscore=b, score_drop=drop, requires_independent_repeat=True)


def repeat_agrees(first, second):
    return (first.get('eligible') is True and second.get('eligible') is True
            and first.get('classification')==second.get('classification')
            and bool(set(first['reasons']) & set(second['reasons'])))
