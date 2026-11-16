"""Post-hoc single-case causal intervention, NOT a general detection fix."""
import copy

BLOCKED_NEURAL_IDS=(92,)

def filter_report(report,neural_ids):
    removed=[];kept=[]
    for obstacle in report['obstacles']:
        index=obstacle['candidate_index']
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(neural_ids):
            raise ValueError('Filter must run on neural candidates BEFORE external obstacle memory')
        if neural_ids[index] in BLOCKED_NEURAL_IDS:
            removed.append(dict(candidate_index=index,neural_track_id=neural_ids[index],
                evidence_track_id=obstacle.get('evidence_track_id'),score=obstacle['raw_score'],box=obstacle['box']))
        else:kept.append(obstacle)
    result=copy.deepcopy(report)
    result['obstacles']=copy.deepcopy(kept)
    if removed:
        conflicts=[o for o in kept if o['state']!='network_retained' and o.get('path_clearance_m',float('inf'))<=0]
        result.update(uncertain_path_conflicts=len(conflicts),action='caution' if conflicts else 'unchanged',
            speed_limit_mps=report['config']['caution_speed_mps'] if conflicts else None)
    result['lineage92_counterfactual']=dict(policy='posthoc_selected_neural_track_id92_only',
        blocked_ids=list(BLOCKED_NEURAL_IDS),removed=removed,before_external_memory=True,
        generalizable_fix=False,raw_neural_outputs_modified=False)
    return result
