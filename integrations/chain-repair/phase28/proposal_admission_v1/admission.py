"""External geometry admission only; no neural state or score modifications."""
import copy,math
from screening import center_visible

def apply(report,evidence,neural_ids,info):
    if evidence['timestamp']!=float(info['timestamp']) or evidence['scene_token']!=info['scene_token']:
        raise ValueError('Stale proposal evidence')
    indexed={row['index']:row for row in evidence['queries']}
    if len(indexed)!=len(evidence['queries']):raise ValueError('Duplicate evidence indices')
    result=copy.deepcopy(report);kept=[];quarantined=[]
    for obstacle in report['obstacles']:
        index=obstacle['candidate_index']
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(neural_ids):
            raise ValueError('Admission must run BEFORE external obstacle memory')
        row=indexed.get(index)
        if row is not None and int(neural_ids[index])!=row['neural_id']:raise ValueError('Query identity mapping mismatch')
        invalid=row is not None and row['untrusted_geometry'] and not row['protected_observation'] and not center_visible(obstacle['box'],info)
        if invalid:
            quarantined.append(dict(candidate_index=index,neural_id=row['neural_id'],label=obstacle['label'],score=obstacle['raw_score'],box=copy.deepcopy(obstacle['box']),
                reason='unobserved_history_motion_inconsistent',motion_innovation_m=row['motion_innovation_m'],history_observations=row['history_observations'],
                history_seconds=row['history_seconds'],geometry_existence_certified=False))
        else:kept.append(copy.deepcopy(obstacle))
    result['obstacles']=kept
    if quarantined:
        # Invalid geometry is not evidence of free space. Retain independent
        # occupancy/depth/unknown constraints and the shared caution speed cap.
        cap=float(report['config']['caution_speed_mps']);old=report.get('speed_limit_mps')
        result['speed_limit_mps']=min(cap,float(old)) if old is not None else cap
        if result.get('action')!='stop':result['action']='caution'
        result['uncertain_path_conflicts']=sum(o['state']!='network_retained' and o.get('path_clearance_m',math.inf)<=0 for o in kept)
    result['proposal_admission']=dict(policy='external_unobserved_geometry_evidence_v1',quarantined=quarantined,before_external_memory=True,
        raw_neural_outputs_modified=False,neural_state_modified=False,free_space_asserted=False,confirmed_driver_memory_cleared=False)
    return result
