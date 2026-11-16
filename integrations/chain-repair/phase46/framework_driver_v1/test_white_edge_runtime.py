from bootstrap import *
import unittest,copy,types,numpy as np
from unittest.mock import patch
from white_edge import WhiteEdge,SCHEMA
import test_live_schedule
from run_driver import configure
from runtime_configuration import load_vehicle_config
import empty_controller
class WhiteEdgeRuntimeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):configure()
 def fixture(self):
  info,report,masks=test_live_schedule.LiveScheduleTests().packet(4.,0.,'edge_runtime')
  packet=dict(schema=SCHEMA,status='fresh',timestamp=0.,ground_truth_used=False,edges=[dict(side='right',a=0.,b=1.85,valid_y_range=[-10,50],observed_y_range=[6,45],uncertainty_m=.15)])
  report.update(require_observed_white_edge=True,observed_white_edge=packet);h._mpc_scheduler.anchor=None;h._mpc_scheduler.last_time=None;h._active_recovery_controller=None
  return info,report,masks
 def test_real_driver_admits_only_edge_feasible_plans(self):
  info,report,masks=self.fixture();v=load_vehicle_config();plan=np.c_[np.linspace(.1,4.,6),np.arange(1,7)*2.]
  identity=lambda f,x0,**kw:types.SimpleNamespace(x=np.array(x0),success=True,message='Current-constraint admission test',nit=0,nfev=0)
  with patch.object(empty_controller,'minimize',side_effect=identity):result=h.optimize(plan,masks,info,reliability=report,vehicle=v,maxiter=1)
  edge=WhiteEdge(report['observed_white_edge'],v)
  if result.get('selected'):self.assertGreaterEqual(edge.swept_clearance(np.asarray(result['selected']['states'])).min(),-1e-5)
  else:self.assertTrue(result['braking_fallback']['white_edge_feasible'])
  self.assertEqual(result['guard']['predicted_road_constraint_mode'],'soft')
 def test_missing_and_stale_edges_skip_mpc_and_brake(self):
  for mode in ('missing','stale'):
   info,report,masks=self.fixture()
   if mode=='missing':report['observed_white_edge']=dict(schema=SCHEMA,status='unavailable',timestamp=0.,ground_truth_used=False,edges=[])
   else:report['observed_white_edge']['timestamp']=-1.
   with patch.object(empty_controller,'minimize',side_effect=AssertionError('MPC must not run without current edge evidence')):result=h.optimize(np.c_[np.zeros(6),np.arange(1,7)*2.],masks,info,reliability=report,vehicle=load_vehicle_config())
   self.assertIsNone(result['selected']);self.assertEqual(result['reason'],'white_edge_evidence_unavailable');self.assertLessEqual(result['braking_fallback']['action']['acc'],0);self.assertEqual(result['braking_fallback']['mode'],'hold')
if __name__=='__main__':unittest.main(verbosity=2)
