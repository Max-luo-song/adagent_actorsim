"""Scene-independent stopping budget, joint braking/steering and recovery."""
import numpy as np
from runtime_configuration import load_vehicle_config

def speed_budget(initial,vehicle,temporal,reaction=.2):
 # Coverage is direction independent. Obstacles remain enforced on each MPC
 # candidate; a blocked neural reference cannot prohibit an alternative bypass.
 center=np.asarray(initial[:2])
 if getattr(temporal,'retained',None):center=center@temporal.matrix+temporal.translation
 distance=max(0.,50.-float(np.max(np.abs(center)))-np.hypot(vehicle.half_width_m,vehicle.half_length_m)-vehicle.clearance_margin_m-float(getattr(temporal,'extra',0.)))
 braking=vehicle.emergency_braking_mps2
 if not np.isfinite([reaction,distance]).all() or reaction<0:raise ValueError('Invalid stopping budget')
 cap=max(0.,np.sqrt((braking*reaction)**2+2*braking*distance)-braking*reaction)
 return min(cap,vehicle.max_speed_mps),{'forecast_coverage_distance_m':distance,'reaction_s':reaction,'braking_mps2':braking,'speed_limit_mps':min(cap,vehicle.max_speed_mps),'source':'perception grid coverage, not an obstacle-free corridor certificate','candidate_obstacle_constraints_preserved':True,'formula':'v*tau+v^2/(2*b)<=coverage_distance'}

def braking_plan(initial,reference,vehicle,clear,road,forecast_available=True,risk_cost=None):
 from empty_controller import integrate_state
 initial=np.asarray(initial);limit=np.deg2rad(vehicle.max_steer_rate_degrees);angle=np.deg2rad(vehicle.max_steer_degrees);reference=np.asarray(reference);target_yaw=float(np.arctan2(reference[min(1,len(reference)-1),0]-initial[0],reference[min(1,len(reference)-1),1]-initial[1])) if len(reference) else 0.
 rows=[];count=max(int(round(6./vehicle.dt)),int(np.ceil(initial[3]/vehicle.emergency_braking_mps2/vehicle.dt)))
 for mode in ['hold','unwind','reference','left','right']:
  states=[initial.copy()];actions=[]
  for _ in range(count):
   state=states[-1];target=state[4] if mode=='hold' else (0. if mode=='unwind' else (np.arctan2(2*vehicle.wheelbase_m*np.sin(target_yaw-state[2]),max(vehicle.wheelbase_m,float(np.linalg.norm(reference[min(1,len(reference)-1)]-state[:2])))) if mode=='reference' else (-angle if mode=='left' else angle)));from native_contract import CONTRACT
   lateral=CONTRACT['comfort_boundaries']['max_abs_lat_accel'];yaw_limit=CONTRACT['comfort_boundaries']['max_abs_yaw_rate'];coupled=min(angle,np.arctan(lateral*vehicle.wheelbase_m/max(state[3]**2,1e-9)),np.arctan(yaw_limit*vehicle.wheelbase_m/max(state[3],1e-9)));target=float(np.clip(target,-coupled,coupled)) if mode!='hold' else target
   rate=float(np.clip((target-state[4])/vehicle.dt,-limit,limit)) if state[3]>1e-8 else 0.;acc=-min(vehicle.emergency_braking_mps2,state[3]/vehicle.dt);actions.append([acc,rate]);states.append(integrate_state(state,[acc,rate],vehicle.dt,vehicle.wheelbase_m))
  states=np.asarray(states);minimum=float(np.min(clear(states)));feasible=bool(minimum>=-1e-5);road_cost=float(np.mean(np.minimum(road.clearance(states),0.)**2)) if road is not None and np.any(road.mask) else 0.;heading=float((states[-1,2]-target_yaw)**2)
  # Without a current raster, permit steering changes only with road support
  # and a known-geometry advantage; no empty-space certificate is invented.
  risk=float(risk_cost(states)) if risk_cost is not None else 0.
  if not np.isfinite(risk) or risk<0:raise ValueError('Invalid braking risk cost')
  row={'mode':mode,'states':states.tolist(),'actions':actions,'action':{'acc':float(actions[0][0]),'steer_rate':float(actions[0][1])},'minimum_known_constraint':minimum if np.isfinite(minimum) else None,'known_geometry_feasible':feasible,'road_cost':road_cost,'risk_cost':risk,'forecast_available':bool(forecast_available),'physical_collision_free_certified':False}
  edge=getattr(road,'white_edge',None)
  edge_gap=float(edge.swept_clearance(states).min()) if edge is not None else float('inf')
  row['white_edge_clearance_m']=edge_gap if np.isfinite(edge_gap) else None
  row['white_edge_feasible']=bool(edge_gap>=-1e-5)
  row['_rank']=(edge_gap < -1e-5,max(0.,-edge_gap),not feasible,0. if feasible else -minimum,road_cost+risk,heading,mode!='hold');rows.append(row)
 allowed=rows
 if not forecast_available:
  hold=rows[0];allowed=[r for r in rows if r['mode']=='hold' or (road is not None and np.any(road.mask) and r['known_geometry_feasible'] and (not hold['known_geometry_feasible'] or r['road_cost']<hold['road_cost']-1e-9))]
 chosen=min(allowed,key=lambda row:row['_rank']).copy();chosen.pop('_rank');chosen['policy']='joint_braking_steering_v1';return chosen

class RecoveryController:
 def __init__(self):self.state='normal';self.clear_since=None;self.clear_count=0;self.previous_time=None;self.continuation_allowed=True
 def speed_cap(self,report):return None
 def release_ready(self,timestamp,config):
  seconds=float(config.get('confirmation_seconds',.2));count=int(config.get('confirmation_observations',3));gap=float(config.get('max_gap_seconds',.75))
  return self.clear_since is not None and self.previous_time is not None and 0<float(timestamp)-self.previous_time<=gap and self.clear_count+1>=count and float(timestamp)-self.clear_since>=seconds-1e-6
 def update(self,optimized,speed,timestamp,dt,braking):
  cfg=optimized.get('guard',{}).get('recovery',{});seconds=float(cfg.get('confirmation_seconds',.2));count=int(cfg.get('confirmation_observations',3));gap=float(cfg.get('max_gap_seconds',.75));timestamp=float(timestamp)
  if not np.isfinite([speed,timestamp,dt,braking,seconds,gap]).all() or speed<0 or min(dt,braking,seconds,gap)<=0 or count<1:raise ValueError('Invalid recovery context')
  if self.previous_time is not None and (timestamp<=self.previous_time or timestamp-self.previous_time>gap):self.state='braking';self.clear_since=None;self.clear_count=0;self.continuation_allowed=False
  self.previous_time=timestamp;before=self.state;selected=optimized.get('selected');source=optimized.get('planning_input_source','fresh_model_forecast');retained=source=='retained_model_forecast';fallback=optimized.get('braking_fallback');action=(fallback or {}).get('action',{'acc':-min(braking,speed/dt),'steer_rate':0.});reason='no_verified_plan'
  if selected is not None:
   controls=np.asarray(selected['controls'],dtype=float)
   if controls.shape!=(6,2) or not np.isfinite(controls).all():raise ValueError('Invalid feasible controls')
  first=np.asarray(selected.get('execution_action',selected['controls'][0]),dtype=float) if selected is not None else None
  if first is not None and (first.shape!=(2,) or not np.isfinite(first).all()):raise ValueError('Invalid current execution action')
  verified=selected is not None and selected.get('feasible') is True and np.isfinite(selected.get('minimum_constraint',np.nan)) and selected['minimum_constraint']>=-1e-5
  if not verified:
   self.state='braking' if speed>1e-8 else 'stopped_reassessing';self.clear_since=None;self.clear_count=0;reason=optimized.get('reason',reason)
  elif retained:
   self.clear_since=None;self.clear_count=0
   if self.continuation_allowed and before in ['normal','caution','degraded','recovering'] and first[0]<=1e-5:self.state='degraded';action={'acc':float(first[0]),'steer_rate':float(first[1])};reason='leased_forecast_continuation'
   else:self.state='braking' if speed>1e-8 else 'stopped_reassessing';reason='fresh_evidence_required_for_release'
  elif before in ['braking','stopped_reassessing','degraded','recovering']:
   if self.clear_since is None:self.clear_since=timestamp
   self.clear_count+=1
   if self.clear_count>=count and timestamp-self.clear_since>=seconds-1e-6:self.state='normal';action={'acc':float(first[0]),'steer_rate':float(first[1])};reason='sustained_fresh_recovery'
   elif before in ['degraded','recovering'] and first[0]<=1e-5:self.state='recovering';action={'acc':float(first[0]),'steer_rate':float(first[1])};reason='bounded_recovery_continuation'
   else:self.state='braking' if speed>1e-8 else 'stopped_reassessing';reason='awaiting_sustained_feasibility'
  else:self.state='normal';action={'acc':float(first[0]),'steer_rate':float(first[1])};reason='feasible_plan'
  self.continuation_allowed=self.state in ['normal','caution','degraded','recovering']
  if not np.isfinite([action['acc'],action['steer_rate']]).all():raise ValueError('Nonfinite chosen action')
  if self.state in ['braking','stopped_reassessing'] and (action['acc']>1e-8 or action['acc']< -braking-1e-8):raise ValueError('Invalid braking fallback action')
  return action,{'policy':'freshness_and_joint_fallback_v1','state_before':before,'state_after':self.state,'reason':reason,'clear_observations':self.clear_count,'clear_since':self.clear_since,'optimization_recomputed':not bool((selected or {}).get('continuation')),'input_source':source,'fallback_used':self.state in ['braking','stopped_reassessing']}
