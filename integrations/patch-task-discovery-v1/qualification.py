"""Separate difficulty, solvability, local repair and held-out generalization."""
from evidence import score


def qualify_task(original,edited,edit_audit,feasibility,repeat):
    """Evidence objects must be bound to the same frozen scene/edit upstream."""
    checks={
        'edit_legal':edit_audit.get('passed') is True,
        'edit_has_effect':edit_audit.get('matched_prefix_and_camera_effect') is True,
        'original_complete':original.get('status')=='completed' and not original.get('any_anomaly'),
        'native_is_real_baseline':edited.get('comparison_mode')=='off',
        'no_infrastructure_failure':edited.get('status') in ('completed','terminated') and not edited.get('error_stage'),
        'reproduced':repeat.get('matched_world_and_seed') is True and repeat.get('same_failure_mechanism') is True,
        'solvability_evidence':feasibility.get('same_world') is True and feasibility.get('legal_dynamics') is True and feasibility.get('road_and_collision_passed') is True,
        'no_online_truth_leak':feasibility.get('given_to_online_policy') is False,
    }
    a,b=score(original),score(edited)
    difficult=(edited.get('termination_reason')=='no_jointly_verified_action'
               or any(edited.get('anomalies',{}).get(k) for k in ('environment_collision','native_route_deviation','abnormal_stall','tracker_error_over_1m'))
               or (a is not None and b is not None and a-b>=.01))
    checks['performance_gap_or_control_failure']=bool(difficult)
    # Score loss alone yields an optimization opportunity, not a physical-failure assertion.
    return dict(ready_for_patch_attempt=all(checks.values()),checks=checks,
                classification='reproducible_solvable_edit_challenge' if all(checks.values()) else 'evidence_pending_or_invalid',
                improvement_guaranteed=False,global_patch_validated=False)


def generalization_verdict(target_pair,original_pair,held_out_pairs,required_cases):
    local=target_pair.get('improvement_demonstrated') is True
    original=original_pair.get('valid') is True and original_pair.get('complete_horizons_comparable') is True and original_pair.get('hd_gain',-1)>=-1e-9
    missing=[c for c in required_cases if c not in held_out_pairs or not held_out_pairs[c].get('valid') or not held_out_pairs[c].get('complete_horizons_comparable')]
    regressions=[c for c in required_cases if c in held_out_pairs and held_out_pairs[c].get('valid') and (held_out_pairs[c].get('hd_gain') is None or held_out_pairs[c]['hd_gain']<-1e-9)]
    return dict(local_improvement=local,original_nonregression=original,missing_held_out_cases=missing,held_out_regressions=regressions,
                registered_generalization_passed=local and original and not missing and not regressions,
                universal_guarantee=False,held_out_used_for_patch_selection=False)
