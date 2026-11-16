"""Remove legacy fixed caution caps; apply explicit confidence speed budgets."""
from evidence_guard import PlanningGuard as EvidenceGuard
class PlanningGuard(EvidenceGuard):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.reported_confidence_cap=args[0].get('confidence_speed_limit_mps') if args else kwargs.get('report',{}).get('confidence_speed_limit_mps')
  if self.reported_confidence_cap is not None and (not __import__('numpy').isfinite(self.reported_confidence_cap) or self.reported_confidence_cap<0):raise ValueError('Invalid confidence speed cap')
  self.previous_strategy_cap=self.speed_limit
  self.speed_limit=(None if self.reported_confidence_cap is None else min(self.vehicle.max_speed_mps,float(self.reported_confidence_cap)))
  # The original controller also re-applies caution_speed for epistemic risk.
  # Its ceiling is the unchanged vehicle physical bound in this diagnostic.
  self.caution_speed=self.vehicle.max_speed_mps
 def audit(self):
  result=super().audit()
  result['confidence_speed_limit_mps']=self.speed_limit
  result['speed_policy']=dict(policy='coverage_and_freshness_speed_budget_v1',original_strategy_cap_mps=self.previous_strategy_cap,strategy_cap_disabled=True,physical_max_speed_mps=self.vehicle.max_speed_mps,physical_acceleration_and_steering_bounds_retained=True,hard_and_soft_obstacle_geometry_unchanged=True,neural_speed_reference_unchanged=True)
  return result
