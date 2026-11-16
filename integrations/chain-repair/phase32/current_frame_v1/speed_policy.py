"""User-authorized closed-loop diagnostic: remove strategy speed caps only."""
from evidence_guard import PlanningGuard as EvidenceGuard
class PlanningGuard(EvidenceGuard):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.previous_strategy_cap=self.speed_limit
  self.speed_limit=None
  # The original controller also re-applies caution_speed for epistemic risk.
  # Its ceiling is the unchanged vehicle physical bound in this diagnostic.
  self.caution_speed=self.vehicle.max_speed_mps
 def audit(self):
  result=super().audit()
  result['speed_policy']=dict(policy='no_strategy_caution_cap_diagnostic_v1',original_strategy_cap_mps=self.previous_strategy_cap,strategy_cap_disabled=True,physical_max_speed_mps=self.vehicle.max_speed_mps,physical_acceleration_and_steering_bounds_retained=True,hard_and_soft_obstacle_geometry_unchanged=True,neural_speed_reference_unchanged=True)
  return result
