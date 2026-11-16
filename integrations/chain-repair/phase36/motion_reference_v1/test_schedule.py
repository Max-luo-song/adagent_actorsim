from bootstrap import *
import ast,copy,types,unittest,numpy as np
from mpc_schedule import Scheduler,POLICY,phased_rollout,state_constraints,check_continuation,SearchSchedule
from runtime_configuration import load_vehicle_config
from empty_controller import rollout,integrate_state
from motion_policy import RecoveryController
class ScheduleTests(unittest.TestCase):
 def setUp(self):
  self.v=load_vehicle_config();self.info=dict(timestamp=0.,scene_token='any',explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),ego_velo=4.,ego_steer=0.,model_lidar_frame={'virtual_to_ego':np.eye(4)});self.plan=np.c_[np.zeros(6),np.arange(1,7)*2.];self.controls=np.zeros((6,2));self.states,self.lidar=rollout(self.controls,4.,0.,[0.,0.],self.v.dt,self.v.wheelbase_m,1.)
 def result(self):return dict(selected=dict(controls=self.controls.tolist(),states=self.states.tolist(),control_interval_s=1.,clearance_lower_bound_m=10.,native_alignment_proxy={'road_loss':0.}),planning_input_source='fresh_model_forecast',guard={'control_interpolation':'constant'})
 def current(self,time):
  info=copy.deepcopy(self.info);info['timestamp']=time;info['explicit_l2g_t']=np.array([0.,4*time,0.]);return info
 def scheduler(self):
  s=Scheduler();s.request(self.plan,self.info,self.v,'fresh_model_forecast');s.record(self.result(),self.plan,self.info);return s
 def test_periodic_boundary_and_no_timer_renewal(self):
  s=self.scheduler()
  for t in [.1,.2,.3,.4]:
   req=s.request(self.plan,self.current(t),self.v,'fresh_model_forecast');self.assertTrue(req['requested']);s.record({'selected':{'continuation':True}},self.plan,self.current(t))
  self.assertEqual(s.request(self.plan,self.current(.5),self.v,'fresh_model_forecast')['trigger'],'period_elapsed')
 def test_rotating_frame_and_nonzero_sensor_mount(self):
  info=copy.deepcopy(self.info);mount=np.eye(4);mount[:3,3]=[.3,-.2,1.];info['model_lidar_frame']['virtual_to_ego']=mount;info['explicit_l2g_t']=mount[:3,3].copy();info['ego_steer']=.1
  states,lidar=rollout(self.controls,4.,.1,mount[:2,3],self.v.dt,self.v.wheelbase_m,1.);plan=lidar[5::5][:6];result=self.result();result['selected']['states']=states.tolist();sch=Scheduler();sch.request(plan,info,self.v,'fresh_model_forecast');sch.record(result,plan,info)
  state=states[1];yaw=state[2];c=np.cos(yaw);s=np.sin(yaw);rotation=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
  body_world=np.r_[state[:2],-mount[2,3]]+info['explicit_l2g_t'];current=copy.deepcopy(info);current.update(timestamp=.1,explicit_l2g_r_mat=rotation,explicit_l2g_t=body_world+rotation@mount[:3,3],ego_velo=state[3],ego_steer=state[4])
  query=np.arange(1,7)*.5+.1;points=np.vstack([np.zeros(2),plan]);old=np.c_[np.interp(query,np.arange(7)*.5,points[:,0]),np.interp(query,np.arange(7)*.5,points[:,1]),np.zeros(6)];current_reference=(np.linalg.inv(rotation)@(old+info['explicit_l2g_t']-current['explicit_l2g_t']).T).T[:,:2]
  self.assertTrue(sch.request(current_reference,current,self.v,'fresh_model_forecast')['requested'])
 def test_reference_tracking_source_clock_and_scene_events(self):
  for kind in ['reference','pose','speed','source','recovery','clock','scene']:
   s=self.scheduler();info=self.current(.1);plan=self.plan.copy();source='fresh_model_forecast';state='normal'
   if kind=='reference':plan[:,0]+=1.
   elif kind=='pose':info['explicit_l2g_t'][0]+=1.
   elif kind=='speed':info['ego_velo']+=1.
   elif kind=='source':source='retained_model_forecast'
   elif kind=='recovery':state='braking'
   elif kind=='clock':info['timestamp']=0.
   elif kind=='scene':info['scene_token']='other'
   with self.subTest(kind=kind):self.assertFalse(s.request(plan,info,self.v,source,state)['requested'])
 def test_exact_piecewise_and_linear_phase_actions(self):
  controls=np.c_[np.arange(6)*.1,np.arange(6)*.01]
  for mode in ['constant','linear']:
   states,lidar,actions=phased_rollout(controls,4.,0.,[.2,.1],self.v,.5,.4,mode)
   for i in [0,1,2,7,29]:
    t=.4+i*.1;expected=controls[min(int(np.floor((t+1e-9)/.5)),5)] if mode=='constant' else np.array([np.interp(t,np.arange(6)*.5,controls[:,j]) for j in range(2)])
    np.testing.assert_allclose(actions[i],expected,atol=1e-14)
   np.testing.assert_allclose(states[1],integrate_state(states[0],actions[0],self.v.dt,self.v.wheelbase_m));self.assertEqual(len(states),31)
 def test_continuation_constraint_rows_match_original_ast(self):
  tree=ast.parse((HERE/'empty_controller.py').read_text());node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='constraints'));node.decorator_list=[];rng=np.random.default_rng(8)
  for limit in [None,2.,10.]:
   states=rng.normal(size=(61,5));clear=lambda x:1.-np.abs(x[:,0]);scope=dict(ambiguity=types.SimpleNamespace(active=False),np=np,vehicle=self.v,guard=types.SimpleNamespace(speed_limit=limit),speed=4.,clear=clear,simulate=lambda *a:(states,None),steer=0.,offset=np.zeros(2));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'original-constraints','exec'),scope)
   np.testing.assert_array_equal(state_constraints(states,clear,self.v,4.,limit),scope['constraints'](np.zeros(12)))
 def context(self,clear=None):
  guard=types.SimpleNamespace(speed_limit=None);aligned=types.SimpleNamespace(components=lambda *a:{'road_loss':0.},cost=lambda *a:0.);req=dict(requested=True,phase_s=.1,controls=self.controls,interval_s=1.,interpolation='constant',previous_clearance=10.,previous_road_loss=0.);return req,guard,aligned,clear or (lambda s:np.full(len(s),10.))
 def check(self,req,guard,aligned,clear):return check_continuation(req,np.array([0.,0.,0.,4.,0.]),np.zeros(2),self.v,clear,guard,aligned,np.zeros((12,2)),1.,'constant',POLICY)
 def test_new_obstacle_and_speed_limit_reject_continuation(self):
  for kind in ['obstacle','cap','near','road','bad_action']:
   req,g,a,c=self.context()
   if kind=='obstacle':c=lambda s:np.where(s[:,1]>2.,-1.,10.)
   elif kind=='cap':g.speed_limit=0.
   elif kind=='near':c=lambda s:np.full(len(s),.1)
   elif kind=='road':a.components=lambda *args:{'road_loss':1.}
   elif kind=='bad_action':req['controls']=np.full((6,2),100.)
   with self.subTest(kind=kind):row,audit=self.check(req,g,a,c);self.assertIsNone(row);self.assertEqual(audit['mode'],'full_replan')
 def test_valid_current_checks_admit_exact_next_action(self):
  row,audit=self.check(*self.context());self.assertTrue(row['continuation']);self.assertTrue(audit['full_current_constraints_checked']);self.assertEqual(row['execution_action'],[0.,0.])
 def test_ambiguity_stop_check_rejects_an_otherwise_valid_continuation(self):
  req,g,a,c=self.context();budget=types.SimpleNamespace(active=True,gap=lambda *args:-.3)
  row,audit=check_continuation(req,np.array([0.,0.,0.,4.,0.]),np.zeros(2),self.v,c,g,a,np.zeros((12,2)),1.,'constant',POLICY,ambiguity=budget)
  self.assertIsNone(row);self.assertEqual(audit['trigger'],'current_constraint_failed');self.assertEqual(audit['ambiguity_stop_gap_m'],-.3)
 def test_far_future_small_slack_does_not_skip_current_check(self):
  req,g,a,c=self.context();req['previous_clearance']=.05;c=lambda s:np.where(s[:,1]>10.,.05,10.)
  row,audit=self.check(req,g,a,c);self.assertIsNotNone(row);self.assertAlmostEqual(audit['clearance_m'],.05)
 def test_controller_executes_phase_action_and_retained_cannot_accelerate(self):
  selected=dict(controls=np.zeros((6,2)).tolist(),execution_action=[.5,.1],feasible=True,minimum_constraint=1.)
  action,_=RecoveryController().update(dict(selected=selected,planning_input_source='fresh_model_forecast'),4.,0.,.1,2.);self.assertEqual(action,{'acc':.5,'steer_rate':.1})
  action,_=RecoveryController().update(dict(selected=selected,planning_input_source='retained_model_forecast'),4.,0.,.1,2.);self.assertLessEqual(action['acc'],0.)
 def test_changed_soft_hypothesis_triggers_full_search(self):
  req,g,a,c=self.context();req['previous_soft_risk']=0.;a.guard=types.SimpleNamespace(soft_cost=lambda *args:1.);a.epistemic=types.SimpleNamespace(cost=lambda *args:0.)
  row,audit=self.check(req,g,a,c);self.assertIsNone(row);self.assertEqual(audit['trigger'],'soft_risk_change')
 def test_search_keeps_both_sides_then_refines(self):
  search=SearchSchedule(POLICY,220);seeds=[(name,np.zeros((6,2))) for name in ['warm','coast','escape_left','escape_right','reachable_stop','left_brake']];visited=[]
  for name,seed,cap,stage in search.candidates(seeds):
   visited.append((name,cap,stage));search.observe(dict(seed=name,controls=seed.tolist(),feasible=True,objective=10.,native_alignment_proxy=dict(local_progress_proxy=1.,dac_proxy=1.,ttc_proxy=1.)))
  self.assertIn('escape_left',[r[0] for r in visited]);self.assertIn('escape_right',[r[0] for r in visited]);self.assertEqual(sum(r[2]=='refine' for r in visited),2);self.assertFalse(search.audit['expanded'])
 def test_poor_quality_retries_all_original_starts_at_full_budget(self):
  search=SearchSchedule(POLICY,220);seeds=[(name,np.zeros((6,2))) for name in ['coast','escape_left','escape_right','left_brake']];visited=[]
  for name,seed,cap,stage in search.candidates(seeds):visited.append((name,cap,stage));search.observe(dict(seed=name,controls=seed.tolist(),feasible=False,objective=10.))
  self.assertEqual([(name,cap) for name,cap,stage in visited if stage=='expanded'],[('expanded_'+name,220) for name,_ in seeds]);self.assertTrue(search.audit['expanded'])
if __name__=='__main__':unittest.main(verbosity=2)
