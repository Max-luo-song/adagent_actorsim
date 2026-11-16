"""Risk never supplies evidence of existence. No scene/ID/position exceptions."""
import numpy as np
from mpc_reuse_v1.native_guard import PlanningGuard as PreviousGuard
POLICY='evidence_before_hard_geometry_v1'

class PlanningGuard(PreviousGuard):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  keep=[];demoted=[];hard_index=0
  for row in self.rows:
   if not row['hard_constraint']:continue
   # The inherited guard uses an envelope for every retained, temporally
   # confirmed or last-known-confirmed object. Unconfirmed rows use rectangles.
   # Urgency and a vulnerable class name are risk signals, not corroboration.
   unsupported=(row['geometry']=='reported_rectangle' and row['evidence_state']!='last_known_confirmed')
   keep.append(not unsupported)
   if unsupported:
    demoted.append(hard_index)
    row.update(original_hard_reason=row['reason'],hard_constraint=False,
      reason='unconfirmed_braking_risk' if row['braking_envelope_clearance_m']<=0 else 'unconfirmed_observation_risk',
      hypothesis_retained=True,existence_promoted_by_risk=False)
   hard_index+=1
  self.demoted_candidate_indices=[row['candidate_index'] for row in self.rows if row.get('existence_promoted_by_risk') is False]
  if demoted:
   self.soft_boxes=np.concatenate([self.soft_boxes,self.hard_boxes[demoted].copy()])
   self.soft_margins=np.concatenate([self.soft_margins,self.hard_margins[demoted].copy()])
   mask=np.asarray(keep,dtype=bool)
   for name in ('hard_boxes','hard_margins','hard_envelopes','hard_radii','hard_ages','hard_oriented','hard_heading_ranges'):
    setattr(self,name,getattr(self,name)[mask].copy())
   self.speed_limit=min(self.caution_speed,self.speed_limit) if self.speed_limit is not None else self.caution_speed
 def audit(self):
  result=super().audit()
  result['proposal_admission']=dict(policy=POLICY,risk_only_hard_promotions_disabled=True,
    demoted_candidate_indices=self.demoted_candidate_indices,
    all_unconfirmed_hypotheses_retained=True,unconfirmed_hypotheses='soft risk and reachable caution speed',
    retained_temporally_confirmed_and_confirmed_memory_unchanged=True,
    raw_occupancy_and_observed_depth_unchanged=True,free_space_asserted=False,ground_truth_used=False)
  return result
