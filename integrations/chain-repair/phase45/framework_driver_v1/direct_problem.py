"""Shared current-frame validation: unchanged MPC constraints and dynamics."""
import numpy as np
class PlanningProblem:
 def __init__(self,**kwargs):self.__dict__.update(kwargs)
 def evaluate(self,controls,quality=True):
  controls=np.asarray(controls,dtype=float)
  if controls.shape!=(6,2) or not np.isfinite(controls).all():return None
  bounds=np.asarray(self.bounds);flat=controls.ravel()
  if np.any(flat<bounds[:,0]-1e-8) or np.any(flat>bounds[:,1]+1e-8):return None
  states,lidar=self.simulate(controls,float(self.info['ego_velo']),float(self.info['ego_steer']),self.offset);minimum=float(self.constraints(flat).min())
  if not np.isfinite(states).all() or minimum< -1e-5:return None
  metrics=self.aligned.components(states,lidar,controls);cost=float(self.objective(flat))
  from mpc_schedule import POLICY
  passed=metrics['dac_proxy']>=POLICY['minimum_dac_proxy'] and metrics['local_progress_proxy']>=POLICY['minimum_progress_proxy'] and metrics['comfort_native_kinematic_flag']==1.
  if not np.isfinite(cost) or quality and not passed:return None
  return dict(seed='distilled_policy',solver_success=None,optimization_skipped=True,optimization_skip_scope='network_proposal_current_full_constraint_validation',feasible=True,minimum_constraint=minimum,clearance_lower_bound_m=float(self.clear(states).min()),objective=cost,controls=controls.tolist(),states=states.tolist(),lidar=lidar.tolist(),control_interval_s=self.interval,native_alignment_proxy=metrics,soft_risk_cost=float(self.aligned.guard.soft_cost(states,np.arange(len(states))*self.vehicle.dt)+self.aligned.epistemic.cost(states)),used_verified_seed=False,seed_name_is_final_direction=False,student_direct=True)
def make_problem(opt,reference,occupancy,info,kwargs):
 args=dict(kwargs);args.update(_continuation=None,_check_continuation_only=False,_problem_callback=lambda p:p,_proposal_provider=None)
 return opt.optimize(reference,occupancy,info,**args)
