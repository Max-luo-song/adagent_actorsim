"""Past genuine model forecasts; absence never creates an empty raster."""
import copy
import numpy as np
class ForecastMemory:
 def __init__(self):self.saved=None;self.previous=None;self.scene=None
 def resolve(self,occupancy,info,report):
  now=float(info['timestamp']);scene=(info['scene_token'],report.get('declared_static_epoch'));cfg=report.get('config',{});lease=float(cfg.get('max_gap_seconds',.75))
  rotation=np.asarray(info['explicit_l2g_r_mat'],dtype=float);translation=np.asarray(info['explicit_l2g_t'],dtype=float)
  if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5):raise ValueError('Invalid forecast coordinate frame')
  if not np.isfinite([now,lease]).all() or lease<=0:raise ValueError('Invalid forecast lease')
  if scene!=self.scene or (self.previous is not None and (now<=self.previous or now-self.previous>lease)):self.saved=None
  self.scene=scene;self.previous=now;contract=info['occupancy_contract'];result=copy.deepcopy(report)
  if occupancy is not None:
   if contract['status'] not in ('valid_empty','valid_nonempty'):raise ValueError('Forecast and transport disagree')
   self.saved=(np.array(occupancy,copy=True),copy.deepcopy(info));result['planning_input_source']='fresh_model_forecast';return occupancy,result
  result['planning_input_source']='unavailable'
  if contract.get('reason')!='no_motion_queries':self.saved=None;return None,result
  health=info.get('perception_health',{});depth=result.get('observed_depth_geometry',{})
  healthy=(health.get('schema')=='uniad_frame_health_v1' and abs(float(health.get('timestamp',-1))-now)<1e-8 and all(health.get(k) is True for k in ['inference_completed','zero_motion_queries_verified','plan_finite','candidates_finite','road_output_present']) and depth.get('depth_available') is True and depth.get('ground_available') is True and result.get('predicted_drivable_area') is not None)
  from temporal_context import AgentlessForecast
  if healthy:
   from drivable_area import DrivableArea
   from runtime_configuration import load_vehicle_config
   healthy=bool(np.any(DrivableArea(result['predicted_drivable_area'],load_vehicle_config()).mask))
  if healthy and (self.saved is None or not np.any(self.saved[0]) or now-float(self.saved[1]['timestamp'])>lease):
   result['planning_input_source']='fresh_agentless_foreground';result['agentless_foreground_contract']={'native_skip_reason':'zero_motion_queries','raster_fabricated':False,'physical_scene_asserted_empty':False,'current_observation_health':health};return AgentlessForecast(),result
  if self.saved is None:return None,result
  masks,old=self.saved;age=now-float(old['timestamp'])
  road=result.get('predicted_drivable_area')
  if not 0<age<=lease+1e-9 or road is None:return None,result
  a=np.asarray(info['explicit_l2g_r_mat']).T@np.asarray(old['explicit_l2g_r_mat'])
  if not np.allclose(a[2],[0,0,1],atol=1e-5) or not np.allclose(a[:,2],[0,0,1],atol=1e-5):return None,result
  result['retained_forecast']={'source_timestamp':float(old['timestamp']),'age_s':age,'lease_s':lease,'source_rotation':np.asarray(old['explicit_l2g_r_mat']).tolist(),'source_translation':np.asarray(old['explicit_l2g_t']).tolist(),'source_contract':old['occupancy_contract'],'unobserved_motion_budget_mps':float(result.get('unobserved_motion_budget_mps',30.)),'raw_current_occupancy_replaced':False,'source':'last_genuine_model_output'}
  result['planning_input_source']='retained_model_forecast';return masks.copy(),result

def frame_health(timestamp,query_audit,plan,candidates,road_present):
 return {'schema':'uniad_frame_health_v1','timestamp':float(timestamp),'inference_completed':True,'zero_motion_queries_verified':query_audit.get('motion_query_count')==0 and query_audit.get('no_motion_query') is True,'plan_finite':np.asarray(plan).shape==(6,2) and bool(np.isfinite(plan).all()),'candidates_finite':all(np.isfinite(x).all() for x in candidates),'road_output_present':bool(road_present)}
