
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import unittest,json,pickle,copy,ast,types,textwrap,numpy as np
from unittest.mock import patch
from current_reference import prepare_report,HISTORY_KEYS,ROAD_SOFT_COST_WEIGHT
from runtime_configuration import load_vehicle_config
import empty_controller as controller
SOURCE=CR/'phase31/runs/occupancy_closed551_20261105T140958Z_48dc4599/closed_loop'
class CurrentReferenceTests(unittest.TestCase):
 def test_history_overrides_removed_without_mutating_geometry_or_input(self):
  report=dict(observed_depth_obstacles_xy=[[1.,2.]],global_evidence_receipt={'stage':'after_driver_memory'},config={},predicted_road_constraint_mode='soft')
  for key in HISTORY_KEYS:report[key]=[[999.,999.]]
  before=pickle.dumps(report);new=prepare_report(report)
  for key in HISTORY_KEYS:self.assertNotIn(key,new)
  self.assertEqual(pickle.dumps(report),before);self.assertIs(new['observed_depth_obstacles_xy'],report['observed_depth_obstacles_xy']);self.assertEqual(new['predicted_road_constraint_mode'],'soft')
 def test_actual_driver_anchor_is_only_evaluation_not_control(self):
  source=(HERE/'protocol_harness.py').read_text();a=source.index('        # This anchor remains an evaluation metric only');b=source.index("        reliability['predicted_drivable_area'] = road_contract",a)
  calls=[];report={key:True for key in HISTORY_KEYS};namespace=dict(navigation_anchor=types.SimpleNamespace(update=lambda *a:calls.append(a)),plan=np.zeros((6,2)),info={},reliability=report)
  exec(textwrap.dedent(source[a:b]),namespace)
  self.assertEqual(len(calls),1)
  for key in HISTORY_KEYS:self.assertNotIn(key,namespace['reliability'])
  self.assertNotIn('committed_route.reference(',source);self.assertNotIn('committed_route.commit(',source)
 def test_actual_objective_road_penalty_is_tenfold(self):
  tree=ast.parse((HERE/'empty_controller.py').read_text());node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='objective'));node.decorator_list=[]
  code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'objective-check','exec')
  namespace=dict(road_soft=types.SimpleNamespace(cost=lambda *a:1.),vehicle=types.SimpleNamespace(dt=.1),np=np,simulate=lambda *a:(np.zeros((2,5)),np.zeros((2,2))),speed=0.,steer=0.,offset=np.zeros(2),spatial_guide=None,progress_weight=0.,navigation_reference=np.zeros((0,2)),_forecast_recovery=False,guard=types.SimpleNamespace(soft_cost=lambda *a:0.),epistemic=types.SimpleNamespace(cost=lambda *a:0.),road=types.SimpleNamespace(clearance=lambda *a:np.array([-1.,-2.])),road_hard=False,substeps=5,ROAD_SOFT_COST_WEIGHT=ROAD_SOFT_COST_WEIGHT)
  exec(code,namespace);self.assertEqual(namespace['objective'](np.zeros(12)),10.);self.assertEqual(ROAD_SOFT_COST_WEIGHT,10.)
 def fixture(self):
  f=SOURCE/'uniad_worker/step_083'
  with (f/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
  del obs
  with np.load(f/'perception.npz') as saved:plan=saved['trajectory'].copy();occ=saved['occupancy'].copy()
  return plan,occ,info,_hugsim_expand(json.loads((f/'guard_memory_report.json').read_text()))
 def test_real_optimizer_uses_current_frame_despite_poisoned_history(self):
  from speed_policy import PlanningGuard
  controller.PlanningGuard=PlanningGuard;plan,occ,info,report=self.fixture();before=pickle.dumps((plan,occ,report));report.update(navigation_anchor_reference=np.full((6,2),999.).tolist(),committed_path_center_xy=[[0.,0.],[999.,999.]],spatial_path_tracking=True)
  poisoned=pickle.dumps(report)
  fake=lambda fun,x0,**kw:types.SimpleNamespace(x=np.asarray(x0),success=False,message='Identity seed for interface regression')
  with patch.object(controller,'minimize',side_effect=fake):result=controller.optimize(plan,occ,info,reliability=report,vehicle=load_vehicle_config())
  audit=result['guard']['navigation_reference'];np.testing.assert_array_equal(audit['original_neural_reference'],plan);np.testing.assert_array_equal(np.asarray(audit['tracking_reference'])[:6],plan)
  self.assertEqual(audit['source'],'current_frame_neural_trajectory');self.assertFalse(audit['history_navigation_override_used']);self.assertEqual(result['guard']['road_soft_cost_weight'],10.);self.assertEqual(pickle.dumps(report),poisoned)
 def test_warm_seed_still_gets_current_objective_optimization(self):
  from speed_policy import PlanningGuard
  controller.PlanningGuard=PlanningGuard;plan,occ,info,report=self.fixture();plan=np.zeros((6,2));info['ego_velo']=0.;info['ego_steer']=0.
  fake=lambda fun,x0,**kw:types.SimpleNamespace(x=np.asarray(x0),success=False,message='Identity seed for warm regression')
  with patch.object(controller,'minimize',side_effect=fake) as call:result=controller.optimize(plan,occ,info,warm=np.zeros((6,2)),reliability=report,vehicle=load_vehicle_config(),_reuse_verified_warm=True)
  self.assertGreater(call.call_count,0);self.assertTrue(all(not t['optimization_skipped'] for t in result['trials']))
 def test_frozen_worker_and_perception_history_unchanged(self):
  old=CR/'phase31/occupancy_closed_v2'
  for name in ('worker.py','protocol_branch_worker.py','occupancy_protocol.py','depth_consistency.py','speed_policy.py','evidence_guard.py'):self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes(),name)
if __name__=='__main__':unittest.main(verbosity=2)
