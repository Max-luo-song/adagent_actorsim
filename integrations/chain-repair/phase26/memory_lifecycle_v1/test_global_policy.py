from bootstrap import *
import unittest,copy,json,ast,types,numpy as np
from forecast_memory import ForecastMemory
from occupancy_protocol import encode,unavailable_result
from temporal_context import TemporalOccupancy
from empty_native_temporal import TemporalOccupancy as BaseTemporal
from motion_policy import RecoveryController,braking_plan,speed_budget
from runtime_configuration import load_vehicle_config

class GlobalPolicyTests(unittest.TestCase):
 def setUp(self):
  self.v=load_vehicle_config();self.masks=np.zeros((1,5,200,200),dtype=np.int64);self.info={'timestamp':0.,'scene_token':'arbitrary_scene','explicit_l2g_r_mat':np.eye(3),'explicit_l2g_t':np.zeros(3),'ego_velo':4.,'ego_steer':.2,'model_lidar_frame':{'virtual_to_ego':np.eye(4)},'occupancy_contract':encode(self.masks)[1]};self.report={'config':{'max_gap_seconds':.75},'predicted_drivable_area':{'present':True},'unobserved_motion_budget_mps':0.}
 def missing(self,time):
  info=copy.deepcopy(self.info);info['timestamp']=time;info['occupancy_contract']=encode(None,0)[1];return info
 def test_no_query_retains_actual_output_without_mutation(self):
  memory=ForecastMemory();self.masks[0,:,120,100]=1;self.info['occupancy_contract']=encode(self.masks)[1];original=self.masks.copy();memory.resolve(self.masks,self.info,self.report);effective,report=memory.resolve(None,self.missing(.1),self.report);np.testing.assert_array_equal(effective,original);np.testing.assert_array_equal(self.masks,original);self.assertFalse(report['retained_forecast']['raw_current_occupancy_replaced']);json.dumps(report)
 def test_lease_is_not_renewed_by_repeated_missing(self):
  memory=ForecastMemory();memory.resolve(self.masks,self.info,self.report)
  for t in [.1,.4,.7]:self.assertIsNotNone(memory.resolve(None,self.missing(t),self.report)[0])
  self.assertIsNone(memory.resolve(None,self.missing(.8),self.report)[0])
 def test_clock_and_scene_reset_drop_old_forecast(self):
  for mode in ['clock','scene']:
   memory=ForecastMemory();memory.resolve(self.masks,self.info,self.report);info=self.missing(.1)
   if mode=='clock':info['timestamp']=0.
   else:info['scene_token']='another_scene'
   self.assertIsNone(memory.resolve(None,info,self.report)[0])
 def test_invalid_head_and_missing_road_do_not_continue(self):
  for bad in ['head','road']:
   memory=ForecastMemory();memory.resolve(self.masks,self.info,self.report);info=self.missing(.1);report=copy.deepcopy(self.report)
   if bad=='head':info['occupancy_contract']=encode(None,2)[1]
   else:report['predicted_drivable_area']=None
   self.assertIsNone(memory.resolve(None,info,report)[0])
 def test_retained_geometry_translates_and_ages_query(self):
  self.masks[0,:,120,100]=1;self.info['occupancy_contract']=encode(self.masks)[1];memory=ForecastMemory();memory.resolve(self.masks,self.info,self.report);current=self.missing(.1);current['explicit_l2g_t']=np.array([0.,2.,0.]);effective,report=memory.resolve(None,current,self.report);temporal=TemporalOccupancy(effective,self.v,info=current,retained=report['retained_forecast']);states=np.array([[0.,5.,0.,1.,0.]]);old=states.copy();old[:,1]+=2.;expected=BaseTemporal(self.masks,self.v).clearance(old,[.1]);np.testing.assert_allclose(temporal.clearance(states,[0.]),expected)
 def test_retained_geometry_preserves_current_depth(self):
  memory=ForecastMemory();memory.resolve(self.masks,self.info,self.report);current=self.missing(.1);current['explicit_l2g_t']=np.array([0.,2.,0.]);effective,report=memory.resolve(None,current,self.report);temporal=TemporalOccupancy(effective,self.v,info=current,retained=report['retained_forecast'],observed_points=[[0.,5.]]);direct=BaseTemporal(self.masks,self.v,observed_points=[[0.,5.]]);states=np.array([[0.,4.,0.,1.,0.]]);np.testing.assert_allclose(temporal.clearance(states,[0.]),direct.clearance(states,[0.]));self.assertLess(temporal.clearance(states,[0.])[0],0.)
 def test_current_depth_not_inflated_by_old_forecast_age(self):
  memory=ForecastMemory();report=dict(self.report,unobserved_motion_budget_mps=10.);memory.resolve(self.masks,self.info,report);current=self.missing(.2);masks,r=memory.resolve(None,current,report);temporal=TemporalOccupancy(masks,self.v,info=current,retained=r['retained_forecast'],observed_points=[[0.,5.]]);direct=BaseTemporal(masks,self.v,observed_points=[[0.,5.]]);states=np.array([[0.,2.,0.,1.,0.]]);np.testing.assert_allclose(temporal.clearance(states,[0.]),direct.clearance(states,[0.]));self.assertGreater(temporal.clearance(states,[0.])[0],0.)
 def test_uncertainty_grows_with_age(self):
  memory=ForecastMemory();report=dict(self.report,unobserved_motion_budget_mps=10.);memory.resolve(self.masks,self.info,report);states=np.zeros((1,5));values=[]
  for t in [.1,.2]:
   info=self.missing(t);masks,r=memory.resolve(None,info,report);values.append(TemporalOccupancy(masks,self.v,info=info,retained=r['retained_forecast']).clearance(states,[0.])[0])
  self.assertAlmostEqual(values[0]-values[1],1.)
 def selected(self,acc=0.,source='fresh_model_forecast'):
  return {'selected':{'controls':[[acc,0.]]*6,'feasible':True,'minimum_constraint':0.},'guard':{},'planning_input_source':source}
 def test_alternating_no_query_does_not_force_braking(self):
  controller=RecoveryController();controller.update(self.selected(),4.,0.,.1,2.)
  for i in range(1,21):
   source='retained_model_forecast' if i%2 else 'fresh_model_forecast';action,decision=controller.update(self.selected(source=source),4.,i*.1,.1,2.);self.assertEqual(action['acc'],0.);self.assertIn(controller.state,['degraded','recovering'])
 def test_retained_prediction_cannot_release_a_stop(self):
  controller=RecoveryController();controller.update({'selected':None},0.,0.,.1,2.)
  for i in range(1,6):self.assertEqual(controller.update(self.selected(1.,'retained_model_forecast'),0.,i*.1,.1,2.)[0]['acc'],0.)
 def test_fresh_recovery_requires_sustained_evidence(self):
  controller=RecoveryController();controller.update({'selected':None},0.,0.,.1,2.)
  for t in [.1,.2]:self.assertEqual(controller.update(self.selected(1.),0.,t,.1,2.)[0]['acc'],0.)
  self.assertTrue(controller.release_ready(.3,{}));self.assertEqual(controller.update(self.selected(1.),0.,.3,.1,2.)[0]['acc'],1.)
 def test_missing_cannot_accelerate_even_with_claimed_feasible_plan(self):
  controller=RecoveryController();action,decision=controller.update(self.selected(1.,'retained_model_forecast'),4.,0.,.1,2.);self.assertLessEqual(action['acc'],0.);self.assertEqual(controller.state,'braking')
 def test_confidence_cap_preserves_guard_geometry(self):
  from speed_policy import PlanningGuard
  from evidence_guard import PlanningGuard as Before
  states=np.zeros((2,5));report={'config':{},'obstacles':[],'confidence_speed_limit_mps':7.25};guard=PlanningGuard(report,self.v,states,np.array([0.,.1]));before=Before(report,self.v,states,np.array([0.,.1]));self.assertEqual(guard.speed_limit,7.25);np.testing.assert_array_equal(guard.hard_boxes,before.hard_boxes);np.testing.assert_array_equal(guard.clearance(states,[0.,.1]),before.clearance(states,[0.,.1]))
 def test_stopping_budget_is_geometry_based_not_fixed_speed(self):
  temporal=BaseTemporal(self.masks,self.v);a,_=speed_budget(np.zeros(5),self.v,temporal);state=np.zeros(5);state[1]=40;b,_=speed_budget(state,self.v,temporal);self.assertGreater(a,b);self.assertGreater(a,2.)
 def test_joint_braking_unwinds_when_corridor_supports_it(self):
  road=types.SimpleNamespace(mask=np.ones((1,1)),clearance=lambda states:1.-np.abs(states[:,0]));clear=lambda states:np.full(len(states),10.);initial=np.array([0.,0.,0.,4.,.25]);plan=np.array([[0.,i] for i in range(1,7)]);result=braking_plan(initial,plan,self.v,clear,road);self.assertLess(result['action']['steer_rate'],0.);states=np.asarray(result['states']);self.assertGreaterEqual(states[:,3].min(),-1e-10);self.assertEqual(states[-1,3],0.);self.assertTrue(result['known_geometry_feasible']);self.assertFalse(result['physical_collision_free_certified'])
 def test_fallback_cannot_claim_clearance_when_all_paths_conflict(self):
  result=braking_plan(np.array([0.,0.,0.,2.,.2]),np.array([[0.,1.]]*6),self.v,lambda states:np.full(len(states),-1.),None,forecast_available=False);self.assertFalse(result['known_geometry_feasible']);self.assertLessEqual(result['action']['acc'],0.);self.assertEqual(result['mode'],'hold')
 def test_clear_cache_preserves_math_and_is_epoch_local(self):
  from mpc_perf_v1.exact_memo import Memo
  tree=ast.parse((HERE/'empty_controller.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='clear');calls=[]
  temporal=types.SimpleNamespace(clearance=lambda states,times:(calls.append(len(states)) or np.ones(len(states))))
  def build():
   scope={'memo':Memo(),'np':np,'vehicle':self.v,'temporal_occupancy':temporal,'guard':types.SimpleNamespace(clearance=lambda states,times:np.full(len(states),10.)),'_forecast_recovery':False,'sweep_allowance':lambda states,v:np.zeros(len(states)),'road':None,'road_hard':False};exec(compile(ast.Module(body=[node],type_ignores=[]),'clear-cache','exec'),scope);return scope['clear']
  clear=build();states=np.zeros((2,5));result=clear(states);result[0]=-99.;self.assertTrue(np.all(clear(states)==1.));self.assertEqual(len(calls),1);states[0,0]=1e-12;clear(states);self.assertEqual(len(calls),2);build()(states);self.assertEqual(len(calls),3)
if __name__=='__main__':unittest.main(verbosity=2)
