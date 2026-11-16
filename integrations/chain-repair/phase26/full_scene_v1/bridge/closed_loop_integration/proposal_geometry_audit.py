"""Read-only serialization of the configured guard's actual proposal arrays.

No asset reference, simulator truth, or matching enters this module.
"""
from functools import wraps


def geometry_receipt(guard):
    rows=[]
    for tier in ('hard','soft'):
        boxes=getattr(guard,tier+'_boxes')
        margins=getattr(guard,tier+'_margins')
        evidence=[r for r in guard.rows if bool(r['hard_constraint'])==(tier=='hard') and not r.get('expired_memory_risk_only')]
        if tier=='soft':
            # EvidenceGuard appends demoted hard proposals after original soft
            # proposals, while its audit rows keep their original order.
            evidence=([r for r in evidence if r.get('existence_promoted_by_risk') is not False]
                +[r for r in evidence if r.get('existence_promoted_by_risk') is False])
        if len(boxes)!=len(margins) or len(boxes)!=len(evidence):
            raise ValueError('Proposal audit arrays/evidence disagree')
        for i,(box,margin,row) in enumerate(zip(boxes,margins,evidence)):
            item=dict(candidate_index=row['candidate_index'],tier=tier,
                geometry=row['geometry'],box=box.tolist(),margin_m=float(margin),
                evidence_state=row['evidence_state'],raw_score=row['raw_score'])
            if tier=='hard':
                item.update(disk=bool(guard.hard_envelopes[i]),
                    oriented_envelope=bool(guard.hard_oriented[i]),
                    radius_m=float(guard.hard_radii[i]),
                    motion_age_s=float(guard.hard_ages[i]),
                    heading_range_rad=float(guard.hard_heading_ranges[i]),
                    time_local_motion=getattr(guard,'local_motion',{}).get(i))
            rows.append(item)
    history=getattr(guard,'expired_history',None)
    if history is not None:
        for i,row in enumerate(guard.expired_rows):
            rows.append(dict(candidate_index=row['candidate_index'],tier='history_risk',
                geometry='expired_memory_risk',box=history.hard_boxes[i].tolist(),
                margin_m=float(history.hard_margins[i]),evidence_state=row['evidence_state'],
                raw_score=row['raw_score'],disk=False,expired_memory_risk_only=True,
                missing_seconds=row['missing_seconds'],current_action_backup_retained=True,
                backup_geometry=dict(disk=bool(history.hard_envelopes[i]),
                    radius_m=float(history.hard_radii[i]),motion_age_s=float(history.hard_ages[i]),
                    oriented=bool(history.hard_oriented[i]),heading_range_rad=float(history.hard_heading_ranges[i]))))
    return dict(schema='proposal_geometry_receipt_v1',coordinate_system='model_lidar_xy_right_forward',
        box_origin='bottom',yaw='clockwise',rows=rows,ground_truth_used=False,
        scope='actual proposal arrays; depth/occupancy constraints recorded separately')


def install(guard_class):
    original=guard_class.audit
    if getattr(original,'_proposal_geometry_audit',False):return
    @wraps(original)
    def audit(self,*args,**kwargs):
        result=dict(original(self,*args,**kwargs))
        result['proposal_geometry_receipt']=geometry_receipt(self)
        return result
    audit._proposal_geometry_audit=True
    guard_class.audit=audit
