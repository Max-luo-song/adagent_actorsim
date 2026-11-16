from bootstrap import *
import unittest,copy,types,ast,numpy as np
from forecast_memory import ForecastMemory,frame_health
from temporal_context import AgentlessForecast,AgentlessTemporal
from occupancy_protocol import encode
from runtime_configuration import load_vehicle_config
from drivable_area import SCHEMA,AXES
class AgentlessTests(unittest.TestCase):
 def setUp(self):
  self.v=load_vehicle_config();self.info={'timestamp':1.,'scene_token':'generic','explicit_l2g_r_mat':np.eye(3),'explicit_l2g_t':np.zeros(3),'ego_velo':0.,'ego_steer':0.,'model_lidar_frame':{'virtual_to_ego':np.eye(4)},'occupancy_contract':encode(None,0)[1]};self.info['perception_health']=frame_health(1.,{'motion_query_count':0,'no_motion_query':True},np.zeros((6,2)),[np.empty((0,9))],True);self.report={'config':{},'obstacles':[],'predicted_drivable_area':{'schema':SCHEMA,'axes':AXES,'bounds_xy':[-50,-50,50,50],'mask':np.ones((20,20),dtype=bool).tolist(),'threshold':.5,'source':'frozen_uniad_map_head','ground_truth_used':False},'predicted_road_constraint_mode':'soft','observed_depth_geometry':{'depth_available':True,'ground_available':True}}
 def test_healthy_no_queries_is_structural_not_a_fabricated_raster(self):
  value,report=ForecastMemory().resolve(None,self.info,self.report);self.assertIsInstance(value,AgentlessForecast);self.assertEqual(report['planning_input_source'],'fresh_agentless_foreground');self.assertFalse(report['agentless_foreground_contract']['physical_scene_asserted_empty']);self.assertIsNone(encode(None,0)[0].get('occupancy'))
 def test_missing_independent_observation_or_worker_health_stops(self):
  for mode in ['depth','ground','worker','clock']:
   info=copy.deepcopy(self.info);report=copy.deepcopy(self.report)
   if mode in ['depth','ground']:report['observed_depth_geometry'][mode+'_available']=False
   elif mode=='worker':info['perception_health']['inference_completed']=False
   else:info['perception_health']['timestamp']=0.
   self.assertIsNone(ForecastMemory().resolve(None,info,report)[0])
 def test_no_road_support_does_not_authorize_agentless_motion(self):
  report=copy.deepcopy(self.report);report['predicted_drivable_area']['mask']=np.zeros((20,20),dtype=bool).tolist();self.assertIsNone(ForecastMemory().resolve(None,self.info,report)[0])
 def test_previous_nonempty_agent_forecast_is_not_erased(self):
  memory=ForecastMemory();old=copy.deepcopy(self.info);old['timestamp']=.9;masks=np.zeros((1,5,200,200));masks[0,:,120,100]=1.;old['occupancy_contract']=encode(masks)[1];memory.resolve(masks,old,self.report);effective,report=memory.resolve(None,self.info,self.report);self.assertIsInstance(effective,np.ndarray);self.assertEqual(report['planning_input_source'],'retained_model_forecast');self.assertGreater(np.sum(effective),0.)
 def test_agentless_preserves_complete_current_obstacle_geometry(self):
  from empty_native_temporal import TemporalOccupancy
  kwargs={'observed_points':[[0.,4.]],'observed_point_radius':np.sqrt(3)*.25,'observed_cell_rotation':np.eye(3),'unknown_points':[[3.,3.]],'unknown_point_radius':np.sqrt(3)*.25,'unknown_cell_rotation':np.eye(3)};states=np.array([[0.,0.,0.,1.,0.],[0.,3.,0.,1.,0.],[2.,3.,.4,1.,0.]]);times=np.arange(3)*.5;actual=AgentlessTemporal(self.v,**kwargs).clearance(states,times);expected=TemporalOccupancy(np.zeros((1,5,200,200)),self.v,**kwargs).clearance(states,times);np.testing.assert_allclose(actual,expected);self.assertLess(actual.min(),0.)
 def test_agentless_reaches_actual_optimizer_without_array_surrogate(self):
  from empty_controller import optimize
  value,report=ForecastMemory().resolve(None,self.info,self.report);result=optimize(np.zeros((6,2)),value,self.info,warm=np.zeros((6,2)),reliability=report,vehicle=self.v,maxiter=2);self.assertIsNotNone(result['selected']);self.assertTrue(result['selected']['feasible']);self.assertFalse(result['guard']['occupancy']['raster_fabricated'])
 def test_complete_forecast_does_not_get_degraded_budget_from_soft_risk(self):
  from empty_controller import optimize
  from speed_policy import PlanningGuard
  import empty_controller
  report=copy.deepcopy(self.report);report['planning_input_source']='fresh_model_forecast';report['unknown_visibility_geometry']={'epistemic_only_points_xy':[[0.,0.]],'point_radius_m':float(np.sqrt(3)*.25)};empty_controller.PlanningGuard=PlanningGuard;result=optimize(np.zeros((6,2)),np.zeros((1,5,200,200)),self.info,warm=np.zeros((6,2)),reliability=report,vehicle=self.v,maxiter=2);self.assertNotIn('confidence_speed_budget',result['guard'])
 def test_retained_raster_cannot_corroborate_current_heading(self):
  import empty_controller
  from speed_policy import PlanningGuard
  seen=[]
  class Spy(PlanningGuard):
   def __init__(inner,*args,**kwargs):seen.append(kwargs.get('current_occupancy'));super().__init__(*args,**kwargs)
  previous=empty_controller.PlanningGuard;empty_controller.PlanningGuard=Spy
  try:
   report=copy.deepcopy(self.report);report['retained_forecast']={'source_rotation':np.eye(3).tolist(),'source_translation':[0,0,0],'age_s':.1,'unobserved_motion_budget_mps':0.};report['planning_input_source']='retained_model_forecast';report['confidence_speed_limit_mps']=0.;masks=np.zeros((1,5,200,200));masks[0,:,140,140]=1;result=empty_controller.optimize(np.zeros((6,2)),masks,self.info,warm=np.zeros((6,2)),reliability=report,vehicle=self.v,maxiter=2);self.assertIsNotNone(result['selected']);self.assertIsNone(seen[0])
  finally:empty_controller.PlanningGuard=previous
 def test_identical_starts_reuse_only_the_same_current_problem(self):
  from empty_controller import optimize
  value,report=ForecastMemory().resolve(None,self.info,self.report);result=optimize(np.zeros((6,2)),value,self.info,reliability=report,vehicle=self.v,maxiter=2);self.assertGreater(result['same_problem_seed_cache']['hits'],0);self.assertFalse(result['same_problem_seed_cache']['cross_frame_reuse']);self.assertTrue(all(not row['optimization_skipped'] or row.get('identical_seed_same_problem_reused') for row in result['trials']));self.assertTrue(all(row['minimum_constraint']>=-1e-5 for row in result['trials'] if row['feasible']))
 def test_original_speed_constraint_layout_and_values_are_preserved(self):
  from mpc_perf_v1.exact_memo import Memo
  old=ast.parse((CR/'phase33/native_alignment_v1/empty_controller.py').read_text());new=ast.parse((HERE/'empty_controller.py').read_text());nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='constraints') for tree in [old,new]];rng=np.random.default_rng(7)
  for initial in [0.,4.,20.,35.]:
   for cap in [None,0.,2.,7.5,30.,60.]:
    for moving in [False,True]:
     states=np.zeros((7,5));states[:,3]=rng.uniform(-1.,35.,7) if moving else 0.;results=[]
     for node in nodes:
      scope={'ambiguity':types.SimpleNamespace(active=False),'memo':Memo(),'np':np,'simulate':lambda *args:(states.copy(),None),'speed':initial,'steer':0.,'offset':np.zeros(2),'vehicle':self.v,'guard':types.SimpleNamespace(speed_limit=cap),'clear':lambda states:np.ones(len(states))};exec(compile(ast.Module(body=[node],type_ignores=[]),'speed-equivalence','exec'),scope);results.append(scope['constraints'](np.zeros(12)))
     np.testing.assert_array_equal(results[1],results[0]);self.assertEqual(bool(np.all(results[0]>=-1e-5)),bool(np.all(results[1]>=-1e-5)))
if __name__=='__main__':unittest.main(verbosity=2)
