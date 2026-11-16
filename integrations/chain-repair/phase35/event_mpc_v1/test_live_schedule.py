from bootstrap import *
import copy,types,unittest,numpy as np
from unittest.mock import patch
from runtime_configuration import load_vehicle_config
from occupancy_protocol import encode
from drivable_area import SCHEMA,AXES
from motion_policy import RecoveryController
from run_driver import configure
import empty_controller
class LiveScheduleTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):configure()
 def packet(self,speed,time,scene,obstacle=False):
  masks=np.zeros((1,5,200,200),dtype=np.uint8)
  if obstacle:masks[0,:,116:120,98:102]=1
  info=dict(timestamp=time,scene_token=scene,explicit_l2g_r_mat=np.eye(3),explicit_l2g_t=np.array([0.,speed*time,0.]),ego_velo=speed,ego_steer=0.,model_lidar_frame={'virtual_to_ego':np.eye(4)},occupancy_contract=encode(masks)[1])
  report=dict(config={},obstacles=[],global_evidence_receipt=dict(stage='after_driver_memory',episode=scene,timestamp=time),observed_depth_geometry=dict(depth_available=True,ground_available=True),predicted_drivable_area=dict(schema=SCHEMA,axes=AXES,bounds_xy=[-50,-50,50,50],mask=np.ones((20,20),dtype=bool).tolist(),source='frozen_uniad_map_head',ground_truth_used=False),predicted_road_constraint_mode='soft')
  return info,report,masks
 def identity(self,fun,x0,**kwargs):return types.SimpleNamespace(x=np.asarray(x0).copy(),success=True,message='Identity start; scheduler integration regression',nit=0,nfev=0)
 def test_full_driver_two_plans_and_four_current_checks_at_three_speeds(self):
  v=load_vehicle_config()
  for speed in [2.,4.,8.]:
   h._mpc_scheduler.anchor=None;h._mpc_scheduler.last_time=None;h._active_recovery_controller=RecoveryController();plan=np.c_[np.zeros(6),np.arange(1,7)*.5*speed];warm=None;modes=[];world=np.array([0.,0.,0.,speed,0.])
   with patch.object(empty_controller,'minimize',side_effect=self.identity) as solver:
    for tick in range(6):
     info,report,masks=self.packet(world[3],tick*.1,'speed_'+str(speed));info['explicit_l2g_t']=np.r_[world[:2],0.];c=np.cos(world[2]);s=np.sin(world[2]);info['explicit_l2g_r_mat']=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
     info['ego_steer']=world[4];before=solver.call_count;result=h.optimize(plan,masks,info,reliability=report,vehicle=v,warm=warm,maxiter=1);modes.append(result['guard']['scheduling']['mode']);selected=result['selected'];self.assertIsNotNone(selected);action,decision=h._active_recovery_controller.update(result,world[3],tick*.1,.1,2.);warm=np.asarray(selected['controls'])
     if 0<tick<5:self.assertEqual(solver.call_count,before);self.assertFalse(decision['optimization_recomputed']);self.assertTrue(result['guard']['scheduling']['full_current_constraints_checked'])
     expected=selected.get('execution_action',selected['controls'][0]);np.testing.assert_allclose([action['acc'],action['steer_rate']],expected)
     world=empty_controller.integrate_state(world,[action['acc'],action['steer_rate']],v.dt,v.wheelbase_m)
   self.assertEqual(modes,['full_replan']+['checked_continuation']*4+['full_replan'])
 def test_new_model_obstacle_enters_real_driver_and_forces_full_planning(self):
  v=load_vehicle_config();speed=4.;plan=np.c_[np.zeros(6),np.arange(1,7)*2.];h._mpc_scheduler.anchor=None;h._mpc_scheduler.last_time=None;h._active_recovery_controller=RecoveryController()
  with patch.object(empty_controller,'minimize',side_effect=self.identity) as solver:
   info,report,masks=self.packet(speed,0.,'new_obstacle');first=h.optimize(plan,masks,info,reliability=report,vehicle=v,maxiter=1);before=solver.call_count
   info,report,masks=self.packet(speed,.1,'new_obstacle',obstacle=True);result=h.optimize(plan,masks,info,reliability=report,vehicle=v,maxiter=1)
   self.assertGreater(solver.call_count,before);self.assertEqual(result['guard']['scheduling']['mode'],'full_replan');self.assertEqual(result['guard']['scheduling']['trigger'],'current_constraint_failed');self.assertFalse((result['selected'] or {}).get('continuation',False))
if __name__=='__main__':unittest.main(verbosity=2)
