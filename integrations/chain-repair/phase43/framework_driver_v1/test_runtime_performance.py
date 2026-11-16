from bootstrap import *
import copy,importlib.util,types,unittest,numpy as np
from runtime_configuration import load_vehicle_config
from fast_road import NativeRoadClearance
from reference_geometry import moving_path_error
from scaled_solver import solve
from mpc_schedule import check_continuation,POLICY
class RuntimeTests(unittest.TestCase):
 def road_class(self):
  spec=importlib.util.spec_from_file_location('reference_road11',CR/'phase11/drivable_area.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 def test_native_square_road_matches_reference_with_holes_rotations_and_edges(self):
  ref=self.road_class();v=load_vehicle_config();rng=np.random.default_rng(89)
  for kind in ['holes','all','none']:
   mask=rng.uniform(size=(50,50))>.15 if kind=='holes' else np.full((50,50),kind=='all');road=ref.DrivableArea(dict(schema=ref.SCHEMA,axes=ref.AXES,bounds_xy=[-20,-20,20,20],mask=mask.tolist(),ground_truth_used=False,source='frozen_uniad_map_head'),v);states=np.c_[rng.uniform(-22,22,(250,2)),rng.uniform(-np.pi,np.pi,250),rng.uniform(0,8,250),np.zeros(250)];before=states.copy();np.testing.assert_array_equal(NativeRoadClearance(road)(states),road.clearance(states));np.testing.assert_array_equal(states,before)
 def test_native_road_does_not_accept_invalid_poses(self):
  ref=self.road_class();road=ref.DrivableArea(dict(schema=ref.SCHEMA,axes=ref.AXES,bounds_xy=[-10,-10,10,10],mask=np.ones((10,10),bool).tolist(),ground_truth_used=False,source='frozen_uniad_map_head'),load_vehicle_config())
  with self.assertRaises(ValueError):NativeRoadClearance(road)(np.full((2,5),np.nan))
 def test_geometry_ignores_pure_neural_time_scaling(self):
  a=np.c_[np.zeros(6),np.arange(1,7)];b=a*3.;self.assertAlmostEqual(moving_path_error(a,np.eye(3),np.zeros(3),b,np.eye(3),[0.,.8,0.]),0.)
 def test_geometry_detects_lane_direction_and_reverse_changes(self):
  a=np.c_[np.zeros(6),np.arange(1,7)]
  for b in [a*np.array([1.,-1.]),a+np.c_[np.arange(1,7)*.4,np.zeros(6)]]:self.assertGreater(moving_path_error(a,np.eye(3),np.zeros(3),b,np.eye(3),np.zeros(3)),.4)
 def test_geometry_is_invariant_to_world_yaw_and_translation(self):
  a=np.c_[np.arange(1,7)*.1,np.arange(1,7)];angle=.73;r=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]]);t=np.array([12.,-7.,2.]);b=a.copy();b[:,0]+=.1
  x=moving_path_error(a,np.eye(3),np.zeros(3),b,np.eye(3),[0.,.3,0.]);y=moving_path_error(a,r,t,b,r,t+r@np.array([0.,.3,0.]));self.assertAlmostEqual(x,y,places=12)
 def test_scaling_preserves_physical_inputs_and_result(self):
  recorded={};x=np.array([1.,.1]);bounds=[(-2.,2.),(-.25,.25)]
  def optimizer(fun,z,**kwargs):
   recorded.update(z=z.copy(),bounds=kwargs['bounds'],value=fun(z),constraints=kwargs['constraints'][0]['fun'](z));return types.SimpleNamespace(x=z.copy())
  result=solve(optimizer,lambda a:float(a@a),x,lambda a:np.array([2.-a[0],.25-a[1]]),bounds,{})
  np.testing.assert_array_equal(result.x,x);np.testing.assert_array_equal(recorded['z'],[.5,.4]);np.testing.assert_array_equal(recorded['bounds'],[(-1.,1.),(-1.,1.)]);self.assertEqual(recorded['value'],float(x@x));np.testing.assert_array_equal(recorded['constraints'],[1.,.15])
 def test_continuation_rejects_current_objective_deterioration(self):
  from mpc_schedule import check_continuation
  v=load_vehicle_config();request=dict(requested=True,phase_s=.1,controls=np.zeros((6,2)).tolist(),interval_s=1.,interpolation='constant',previous_clearance=10.,previous_road_loss=0.,previous_objective=10.);guard=types.SimpleNamespace(speed_limit=None);objective=types.SimpleNamespace(components=lambda *args:{'road_loss':0.},cost=lambda *args:20.)
  row,audit=check_continuation(request,np.array([0.,0.,0.,4.,0.]),np.zeros(2),v,lambda states:np.full(len(states),10.),guard,objective,np.zeros((12,2)),1.,'constant',POLICY);self.assertIsNone(row);self.assertEqual(audit['trigger'],'current_objective_changed')
 def test_explicit_neural_stop_still_forces_replanning(self):
  from mpc_schedule import Scheduler
  v=load_vehicle_config();info=dict(timestamp=0.,scene_token='stop_case',ego_velo=4.,ego_steer=0.,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.zeros(3),model_lidar_frame={'virtual_to_ego':np.eye(4)});plan=np.c_[np.zeros(6),np.arange(1,7)*2.]
  from empty_controller import rollout
  controls=np.zeros((6,2));states,lidar=rollout(controls,4.,0.,np.zeros(2),control_interval_s=1.);scheduler=Scheduler();scheduler.request(plan,info,v,'fresh_model_forecast');scheduler.record(dict(selected=dict(controls=controls.tolist(),states=states.tolist(),control_interval_s=1.,clearance_lower_bound_m=10.),planning_input_source='fresh_model_forecast',guard={'control_interpolation':'constant'}),plan,info);info=copy.deepcopy(info);info.update(timestamp=.1,explicit_l2g_t=np.array([0.,.4,0.]));self.assertEqual(scheduler.request(np.zeros((6,2)),info,v,'fresh_model_forecast')['trigger'],'reference_changed')
 def test_scaled_solver_keeps_real_constraint_feasibility(self):
  from scipy.optimize import minimize
  result=solve(minimize,lambda x:float((x[0]-1.)**2+(x[1]-.1)**2),np.zeros(2),lambda x:np.array([.05-x[1],x[0]-.5]),[(-2.,2.),(-.25,.25)],{'maxiter':60,'ftol':1e-9});self.assertGreaterEqual(result.x[0],.5-1e-6);self.assertLessEqual(result.x[1],.05+1e-6)
if __name__=='__main__':unittest.main(verbosity=2)
