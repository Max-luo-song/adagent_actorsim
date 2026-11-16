"""Native metric structure with explicitly perception-based smooth proxies."""
import numpy as np
from native_contract import CONTRACT,comfort_terms,combine
from current_reference import ROAD_SOFT_COST_WEIGHT
from road_priority import road_priority_cost, POLICY as ROAD_PRIORITY_POLICY
class NativeAlignedObjective:
 def __init__(self,plan,vehicle,road_soft,temporal,guard,clear):
  from mpc_perf_v1.exact_memo import Memo
  self.forecast_memo=Memo(capacity=128)
  self.forecast_query=self.forecast_memo.wrap(self._forecast_query)
  self.plan=np.asarray(plan,dtype=float).reshape(6,2).copy();self.v=vehicle;self.road=road_soft;self.temporal=temporal;self.guard=guard;self.clear=clear
  self.stride=int(round(.5/vehicle.dt));self.stop=bool(np.linalg.norm(self.plan[-1])<vehicle.half_length_m)
  path=np.vstack([np.zeros(2),self.plan]);length=np.linalg.norm(np.diff(path,axis=0),axis=1);self.path=path[np.r_[True,length>1e-8]]
  arc=np.r_[0.,np.cumsum(length)];self.expected=float(np.interp(2.,np.arange(7)*.5,arc));self.progress_window_s=2.
 def _forecast_query(self,states,times):
  return np.minimum(self.temporal.clearance(states,times),self.guard.clearance(states,times))
 def progress(self,lidar):
  if self.stop or len(self.path)<2 or self.expected<1e-8:return 1.
  count=min(len(lidar)-1,int(round(self.progress_window_s/self.v.dt)))
  p=np.asarray(lidar)[:count+1];mid=(p[:-1]+p[1:])/2;segments=np.diff(self.path,axis=0);length=np.linalg.norm(segments,axis=1)
  delta=mid[:,None,:]-self.path[None,:-1,:];u=np.clip(np.sum(delta*segments,axis=2)/length**2,0.,1.);closest=self.path[None,:-1,:]+u[:,:,None]*segments
  distance=np.linalg.norm(mid[:,None,:]-closest,axis=2);width=max(self.v.half_width_m,1e-3);z=-.5*(distance/width)**2;z-=z.max(axis=1,keepdims=True);w=np.exp(z);w/=w.sum(axis=1,keepdims=True)
  tangent=w@(segments/length[:,None]);signed=float(np.sum(np.diff(p,axis=0)*tangent));return float(np.clip(signed/self.expected,0.,1.))
 def components(self,states,lidar,controls):
  states=np.asarray(states);lidar=np.asarray(lidar);sample=states[::self.stride];times=np.arange(len(states))*self.v.dt;sample_times=times[::self.stride];margin=max(self.v.clearance_margin_m,.05)
  nc_gap=np.asarray(self.clear(states));nc_excess=float(np.max(np.maximum(-nc_gap,0.))/margin);nc=1./(1.+nc_excess**2)
  norm=max(len(states)/self.stride,1.);road_loss=float(self.road.cost(states,self.stride)/norm) if self.road is not None else 1.
  road_supported=self.road is not None and self.road.has_road
  dac=1./(1.+ROAD_SOFT_COST_WEIGHT*road_loss) if road_supported else 0.
  known=sample_times<=self.progress_window_s+1e-9;nodes=sample[known].copy();query_times=sample_times[known]
  gaps=[]
  if len(nodes)>1:
   velocity=np.diff(nodes[:,:2],axis=0)/.5;velocity=np.vstack([velocity[0],velocity])
   for tau in CONTRACT['ttc_projection_offsets_s']:
    shifted=nodes.copy();shifted[:,:2]+=velocity*tau
    gap=self.forecast_query(shifted,query_times);gaps.append(float(np.min(gap)))
  ttc_gap=min(gaps) if gaps else 0.;ttc_excess=max(-ttc_gap,0.)/margin;ttc=1./(1.+ttc_excess**2)
  comfortable=comfort_terms(sample[:,:3],.5);progress=self.progress(lidar);score=combine(nc,dac,ttc,comfortable['proxy'],progress)
  return dict(road_peak_violation_m2=self.road.peak_violation(states,buffer_m=ROAD_PRIORITY_POLICY['preferred_road_clearance_m'],start_s=ROAD_PRIORITY_POLICY['peak_start_s']) if self.road is not None else 0.,nc_proxy=float(nc),dac_proxy=float(dac),ttc_proxy=float(ttc),comfort_proxy=float(comfortable['proxy']),comfort_native_kinematic_flag=comfortable['native_flag'],comfort_max_normalized_ratio=comfortable['max_normalized_ratio'],local_progress_proxy=progress,pdms_proxy=score['pdms'],local_hd_proxy=score['hdscore'],road_loss=road_loss,road_support_available=road_supported,minimum_ttc_proxy_gap_m=ttc_gap,actual_max_lateral_accel_mps2=float(np.max(np.abs(states[:,3]**2*np.tan(states[:,4])/self.v.wheelbase_m))),proxy_only=True,ground_truth_used=False)
 def cost(self,states,lidar,controls,reference):
  metrics=self.components(states,lidar,controls)
  from spatial_reference import spatial_error,POLICY
  timed=float(np.mean(np.sum((lidar[self.stride::self.stride]-reference)**2,axis=1)))
  spatial=spatial_error(lidar[self.stride::self.stride],self.plan,self.stop)
  desired=0. if self.stop else float(getattr(self,'cruise_preference',states[0,3]))
  speed_error=float(np.mean((np.asarray(states)[1:,3]-desired)**2))
  controls=np.asarray(controls);limits=np.array([max(abs(self.v.min_acc_mps2),self.v.max_acc_mps2),np.deg2rad(self.v.max_steer_rate_degrees)])
  regularization=.01*float(np.mean((controls/limits)**2))+.01*float(np.mean((np.diff(controls,axis=0)/limits)**2))
  return float(100.*(1.-metrics['local_hd_proxy'])+POLICY['spatial_tracking_weight']*spatial+POLICY['cruise_speed_weight']*speed_error+POLICY['neural_timing_weight']*timed+regularization+self.guard.soft_cost(states,np.arange(len(states))*self.v.dt)+self.epistemic.cost(states)+road_priority_cost(metrics))
 def audit(self):
  return dict(road_priority_policy=dict(ROAD_PRIORITY_POLICY),reference_speed_policy='current_geometry_soft_cruise_v1',cruise_preference_mps=getattr(self,'cruise_preference',None),cruise_is_speed_cap=False,neural_stop_preserved=True,policy='native_metric_aligned_perception_proxy_v1',native_source_sha256=CONTRACT['source_sha256'],native_weights=CONTRACT['active_weights'],aggregation=CONTRACT['aggregation'],proxy_only=True,ground_truth_used=False,local_progress_is_not_native_route_completion=True,progress_window_s=self.progress_window_s,ttc_offsets_s=CONTRACT['ttc_projection_offsets_s'],ttc_forecast_window_s=2.,comfort_sampling_s=.5,native_lateral_acceleration_implementation=CONTRACT['native_lat_acceleration_implementation'],case_specific_goal=False)
