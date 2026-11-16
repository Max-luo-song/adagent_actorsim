"""Event scheduling; current observations and hard constraints are always checked."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import copy,json
pass  # Path is provided by the portable boundary above
import numpy as np
POLICY=_hugsim_expand(json.loads(Path(__file__).with_name('scheduling_policy.json').read_text()))

class Scheduler:
 def __init__(self,policy=None):self.policy=dict(POLICY if policy is None else policy);self.anchor=None;self.last_time=None
 def elapsed(self,info):return None if self.anchor is None else float(info['timestamp'])-self.anchor['time']
 def request(self,reference,info,vehicle,source,controller_state='normal'):
  now=float(info['timestamp']);a=self.anchor;reason=None
  if self.last_time is not None and (now<=self.last_time or now-self.last_time>vehicle.dt+1e-6):reason='clock_discontinuity';self.anchor=None;a=None
  self.last_time=now
  if not self.policy.get('enabled',True):reason='scheduling_disabled'
  if a is None:reason=reason or 'no_previous_plan'
  elif controller_state not in ('normal','caution'):reason='recovery_state'
  elif source!=a['source'] or source not in ('fresh_model_forecast','fresh_agentless_foreground'):reason='perception_source'
  elif info.get('scene_token')!=a['scene'] or info.get('declared_static_epoch')!=a['epoch']:reason='scene_or_epoch'
  else:
   age=now-a['time'];k=int(round(age/vehicle.dt))
   if age>=self.policy['replan_period_s']-1e-8:reason='period_elapsed'
   elif age<=0 or abs(age-k*vehicle.dt)>1e-6 or k>=len(a['states']):reason='invalid_phase'
   else:
    r=np.asarray(info['explicit_l2g_r_mat']);t=np.asarray(info['explicit_l2g_t']);mount=np.asarray(info['model_lidar_frame']['virtual_to_ego']);off=mount[:2,3];state=a['states'][k]
    expected=np.r_[state[:2],-a['mount'][2,3]]@a['r'].T+a['t'];actual=np.r_[-off,-mount[2,3]]@r.T+t
    forward=np.r_[np.sin(state[2]),np.cos(state[2]),0.]@a['r'].T@r
    yaw=np.arctan2(forward[0],forward[1]);position=float(np.linalg.norm(expected-actual))
    if position>vehicle.clearance_margin_m*self.policy['tracking_position_margin_factor'] or abs(yaw)>self.policy['tracking_yaw_rad'] or abs(state[3]-info['ego_velo'])>self.policy['tracking_speed_mps'] or abs(state[4]-info['ego_steer'])>np.deg2rad(vehicle.max_steer_rate_degrees)*vehicle.dt+1e-6:reason='tracking_error'
    else:
     query=np.arange(1,7)*.5+age;known=query<=3.+1e-9;path=np.vstack([np.zeros(2),a['reference']]);xy=np.stack([np.interp(query[known],np.arange(7)*.5,path[:,j]) for j in range(2)],axis=1)
     world=np.c_[xy,np.zeros(len(xy))]@a['r'].T+a['t'];current=(world-t)@r
     error=float(np.max(np.linalg.norm(current[:,:2]-np.asarray(reference)[known],axis=1))) if known.any() else np.inf
     old_stop=np.linalg.norm(a['reference'][-1])<vehicle.half_length_m;new_stop=np.linalg.norm(np.asarray(reference)[-1])<vehicle.half_length_m
     if old_stop!=new_stop:error=float('inf')
     elif not old_stop and self.policy.get('reference_metric')=='spatial_arc':
      from reference_geometry import moving_path_error
      error=moving_path_error(a['reference'],a['r'],a['t'],reference,r,t)
     if error>vehicle.half_width_m*self.policy['reference_change_half_width_fraction']:reason='reference_changed'
  if reason:return {'requested':False,'trigger':reason}
  return {'requested':True,'trigger':'between_replans','phase_s':age,'controls':a['controls'].tolist(),'interval_s':a['interval'],'interpolation':a['interpolation'],'previous_clearance':a['clearance'],'previous_road_loss':a['road_loss'],'previous_soft_risk':a['soft_risk'],'previous_objective':a.get('objective'),'reference_error_m':error,'reference_metric':self.policy.get('reference_metric','timed')}
 def record(self,result,reference,info):
  selected=result.get('selected')
  if selected is None:self.anchor=None;return
  if selected.get('continuation'):return
  self.anchor=dict(objective=selected.get('objective'),time=float(info['timestamp']),states=np.asarray(selected['states']),controls=np.asarray(selected['controls']),interval=selected['control_interval_s'],interpolation=result.get('guard',{}).get('control_interpolation','constant'),reference=np.asarray(reference).copy(),r=np.asarray(info['explicit_l2g_r_mat']).copy(),t=np.asarray(info['explicit_l2g_t']).copy(),mount=np.asarray(info['model_lidar_frame']['virtual_to_ego']).copy(),source=result.get('planning_input_source'),scene=info.get('scene_token'),epoch=info.get('declared_static_epoch'),clearance=selected['clearance_lower_bound_m'],road_loss=selected.get('native_alignment_proxy',{}).get('road_loss',0.),soft_risk=selected.get('soft_risk_cost'))


def phased_rollout(controls,speed,steer,offset,vehicle,interval,phase,interpolation):
 from empty_controller import integrate_state
 controls=np.asarray(controls,dtype=float).reshape(6,2);offset=np.asarray(offset);state=np.r_[-offset,0.,speed,steer];states=[state.copy()];lidar=[np.zeros(2)];actions=[]
 for tick in range(int(round(6*interval/vehicle.dt))):
  time=phase+tick*vehicle.dt
  if interpolation=='linear':action=np.array([np.interp(time,np.arange(6)*interval,controls[:,j]) for j in range(2)])
  else:action=controls[min(int(np.floor((time+1e-9)/interval)),5)].copy()
  actions.append(action);state=integrate_state(state,action,vehicle.dt,vehicle.wheelbase_m);states.append(state.copy());c=np.cos(state[2]);s=np.sin(state[2]);lidar.append(state[:2]+[c*offset[0]+s*offset[1],-s*offset[0]+c*offset[1]])
 return np.asarray(states),np.asarray(lidar),np.asarray(actions)


def state_constraints(states,clear,vehicle,speed,speed_limit):
 rows=[clear(states),states[:,3],vehicle.max_speed_mps-states[:,3],np.deg2rad(vehicle.max_steer_degrees)-np.abs(states[:,4])]
 if speed_limit is not None:rows.append(np.maximum(speed_limit,speed-vehicle.emergency_braking_mps2*np.arange(len(states))*vehicle.dt)-states[:,3])
 return np.concatenate(rows)


def check_continuation(request,initial,offset,vehicle,clear,guard,aligned,reference,interval,interpolation,policy,ambiguity=None):
 audit=dict(mode='full_replan',trigger=(request or {}).get('trigger','not_scheduled'),full_current_constraints_checked=False)
 if not request or not request.get('requested'):return None,audit
 phase=float(request['phase_s']);controls=np.asarray(request['controls'],dtype=float)
 if not 0<phase<min(policy['replan_period_s'],interval)-1e-8 or request['interval_s']!=interval or request['interpolation']!=interpolation or controls.shape!=(6,2) or not np.isfinite(controls).all():audit['trigger']='invalid_continuation';return None,audit
 states,lidar,actions=phased_rollout(controls,initial[3],initial[4],offset,vehicle,interval,phase,interpolation)
 residual=float(state_constraints(states,clear,vehicle,initial[3],guard.speed_limit).min());clearance=float(clear(states).min());rate=np.deg2rad(vehicle.max_steer_rate_degrees)
 physical=bool(np.isfinite(states).all() and actions[:,0].min()>=vehicle.min_acc_mps2-1e-8 and actions[:,0].max()<=vehicle.max_acc_mps2+1e-8 and np.abs(actions[:,1]).max()<=rate+1e-8)
 if ambiguity is not None and ambiguity.active:
  gap=ambiguity.gap(initial,actions[0]);residual=min(residual,gap);audit['ambiguity_stop_gap_m']=gap
 audit.update(full_current_constraints_checked=True,minimum_constraint=residual,clearance_m=clearance)
 if not physical or residual< -1e-5:audit['trigger']='current_constraint_failed';return None,audit
 immediate=clear(states[:int(round(policy['replan_period_s']/vehicle.dt))+1]);immediate_gap=float(immediate.min());audit['next_period_clearance_m']=immediate_gap
 if immediate_gap<vehicle.clearance_margin_m*policy['clearance_reserve_margin_factor'] or clearance<request['previous_clearance']-policy['clearance_drop_m']:audit['trigger']='clearance_event';return None,audit
 metrics=aligned.components(states,lidar,controls)
 if metrics['road_loss']>request['previous_road_loss']+policy['road_loss_increase']:audit['trigger']='road_change';return None,audit
 times=np.arange(len(states))*vehicle.dt
 risk=float(aligned.guard.soft_cost(states,times)+aligned.epistemic.cost(states)) if hasattr(aligned,'epistemic') else 0.
 if not np.isfinite(risk) or (request.get('previous_soft_risk') is not None and risk>request['previous_soft_risk']+policy['soft_risk_increase']):audit['trigger']='soft_risk_change';return None,audit
 current_objective=aligned.cost(states,lidar,controls,reference);previous_objective=request.get('previous_objective')
 if not np.isfinite(current_objective):audit['trigger']='nonfinite_current_objective';return None,audit
 if previous_objective is not None and current_objective>previous_objective+policy.get('continuation_objective_increase_fraction',.01)*max(abs(previous_objective),1.):audit.update(trigger='current_objective_changed',current_objective=current_objective,previous_objective=previous_objective);return None,audit
 row=dict(soft_risk_cost=risk,seed='checked_remaining_plan',solver_success=None,optimization_skipped=True,optimization_skip_scope='scheduled_continuation_full_current_verification',identical_seed_same_problem_reused=False,feasible=True,minimum_constraint=residual,clearance_lower_bound_m=clearance,objective=current_objective,controls=controls.tolist(),states=states.tolist(),lidar=lidar.tolist(),control_interval_s=interval,control_phase_s=phase,control_interpolation=interpolation,execution_action=actions[0].tolist(),continuation=True,native_alignment_proxy=metrics,seed_name_is_final_direction=False)
 audit.update(mode='checked_continuation',trigger='current_checks_passed',tail_policy='hold final control only within newly verified horizon')
 return row,audit


class SearchSchedule:
 def __init__(self,policy,maxiter):self.policy=policy;self.maxiter=maxiter;self.rows=[];self.audit={'staged':True,'expanded':False,'global_optimum_claimed':False}
 def observe(self,row):self.rows.append(row)
 def candidates(self,seeds):
  primary_names=['warm','predicted_road_route','coast','escape_left','escape_right','reachable_stop'];primary=[x for name in primary_names for x in seeds if x[0]==name];remaining=[x for x in seeds if x[0] not in primary_names]
  self.audit['primary_seeds']=[name for name,_ in primary]
  for name,seed in primary:yield name,seed,min(self.maxiter,self.policy['coarse_iterations']),'coarse'
  feasible=sorted([r for r in self.rows if r['feasible']],key=lambda r:r['objective']);before=feasible[0]['objective'] if feasible else None
  for r in feasible[:self.policy['refine_count']]:yield 'refine_'+r['seed'],np.asarray(r['controls']),min(self.maxiter,self.policy['refine_iterations']),'refine'
  feasible=sorted([r for r in self.rows if r['feasible']],key=lambda r:r['objective']);best=feasible[0] if feasible else None;m=(best or {}).get('native_alignment_proxy',{});improvement=(before-best['objective'])/max(abs(before),1.) if before is not None and best else None
  quality=bool(best and m.get('local_progress_proxy',0.)>=self.policy['minimum_progress_proxy'] and m.get('dac_proxy',0.)>=self.policy['minimum_dac_proxy'] and m.get('ttc_proxy',0.)>=self.policy['minimum_ttc_proxy'] and improvement is not None and improvement<=self.policy['maximum_refinement_improvement_fraction'])
  self.audit.update(quality_gate_passed=quality,refinement_improvement_fraction=improvement)
  if quality:return
  self.audit['expanded']=True
  # Short probes cannot discard starts: retry every original start at full budget.
  for name,seed in seeds:yield 'expanded_'+name,seed,self.maxiter,'expanded'
