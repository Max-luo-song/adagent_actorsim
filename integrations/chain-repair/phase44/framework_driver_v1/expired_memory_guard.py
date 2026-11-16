"""Expired object anchors are risk, with full geometry retained for action backup.

Fresh proposals, depth, occupancy, physical bounds and margins are unchanged.
No missing observation is declared free. Only the memory lifecycle can retire
an old anchor using repeated reliable whole-box free-space evidence.
"""
import copy
import numpy as np
from evidence_guard import PlanningGuard as BaseGuard

ARRAYS=('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges')


class PlanningGuard(BaseGuard):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  report=args[0] if args else kwargs.get('report',{})
  obstacles={o['candidate_index']:o for o in report.get('obstacles',[])}
  hard=[row for row in self.rows if row['hard_constraint']]
  expired=np.array([row['evidence_state']=='last_known_confirmed'
      and obstacles[row['candidate_index']].get('geometry_evidence_expired') is True
      and obstacles[row['candidate_index']].get('declared_static_epoch') is None for row in hard],dtype=bool)
  self.expired_memory_count=int(expired.sum())
  self.expired_history=copy.copy(self)
  for name in ARRAYS:
   original=getattr(self,name)
   setattr(self.expired_history,name,original[expired].copy())
   setattr(self,name,original[~expired].copy())
  self.expired_rows=[]
  for row,is_expired in zip(hard,expired):
   if not is_expired:continue
   obstacle=obstacles[row['candidate_index']]
   row.update(hard_constraint=False,reason='expired_location_risk_with_full_action_backup',
       expired_memory_risk_only=True,memory_previous_geometry=row['geometry'],
       geometry='expired_memory_risk',missing_seconds=obstacle['missing_seconds'],
       current_action_backup_retained=True)
   self.expired_rows.append(row)

 def full_clearance(self,states,times):
  return np.minimum(BaseGuard.clearance(self,states,times),
      BaseGuard.clearance(self.expired_history,states,times))

 def soft_cost(self,states,times):
  value=super().soft_cost(states,times)
  if not self.expired_memory_count:return value
  gap=BaseGuard.clearance(self.expired_history,states,times)
  penetration=np.maximum(-gap,0.)
  # Unresolved history contributes bounded risk, not another stationary wall.
  return float(value+np.mean((penetration/(1.+penetration))**2))

 def audit(self):
  result=super().audit()
  result['expired_memory']=dict(count=self.expired_memory_count,
      policy='risk_and_current_action_full_geometry_backup',
      current_action_backup_required=bool(self.expired_memory_count),
      future_hard_stationary_anchors_removed=self.expired_memory_count,
      free_space_asserted=False,rows=self.expired_rows,ground_truth_used=False)
  result['proposal_admission']['retained_temporally_confirmed_and_confirmed_memory_unchanged']=not self.expired_memory_count
  return result
