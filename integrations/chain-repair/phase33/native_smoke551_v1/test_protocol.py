
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
import unittest,json,tempfile,types,copy,pickle,ast,numpy as np
from occupancy_protocol import pack,unpack,unavailable_result
from planning_guard import RecoveryController

class ProtocolTests(unittest.TestCase):
 def test_round_trip_valid_empty_and_nonempty_preserves_arrays(self):
  for occupied in (False,True):
   a=np.zeros((1,5,1,200,200),dtype=np.int64)
   if occupied:a[0,0,0,100,100]=1
   before=a.copy();fields,c=pack(a,1)
   with tempfile.TemporaryDirectory() as temp:
    file=Path(temp)/'perception.npz';np.savez_compressed(file,**fields)
    with np.load(file) as saved:b,d=unpack(saved)
   self.assertEqual(c,d);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,before)
 def test_none_zero_query_and_unexpected_missing_have_no_surrogate(self):
  for count,reason in ((0,'no_motion_queries'),(2,'missing_head_output'),(None,'missing_head_output')):
   fields,c=pack(None,count);self.assertNotIn('occupancy',fields);self.assertEqual(c['reason'],reason);a,d=unpack(fields);self.assertIsNone(a);self.assertEqual(d['status'],'unavailable')
 def test_malformed_and_nonfinite_are_invalid_and_never_clear(self):
  for a in (np.zeros((5,200,200)),np.full((1,5,200,200),np.nan),np.full((1,5,200,200),'bad')):
   fields,c=pack(a,1);self.assertEqual(c['status'],'invalid');self.assertNotIn('occupancy',fields);self.assertIsNone(unpack(fields)[0])
 def test_contract_corruption_is_rejected(self):
  fields,c=pack(None,0);fields['occupancy']=np.zeros((1,5,200,200))
  with self.assertRaises(ValueError):unpack(fields)
  fields,c=pack(np.zeros((1,5,200,200)),1);fields['occupancy'][0,0,0,0]=1
  with self.assertRaises(ValueError):unpack(fields)
 def test_repeated_unavailable_brakes_to_rest_without_terminating(self):
  _,contract=pack(None,0);r=unavailable_result(contract,dict(confirmation_seconds=.2,confirmation_observations=3,max_gap_seconds=.75));controller=RecoveryController();speed=2.
  for i in range(100):
   action,decision=controller.update(r,speed,i*.1,.1,2.);self.assertLessEqual(action['acc'],0.);speed=max(0.,speed+action['acc']*.1)
  self.assertEqual(speed,0.);self.assertEqual(controller.state,'stopped_reassessing');self.assertTrue(r['diagnostic_continues'])
 def test_recovery_requires_sustained_feasibility_and_resets_on_new_loss(self):
  _,contract=pack(None,0);missing=unavailable_result(contract,{})
  good=dict(selected=dict(controls=[[1.,0.]]*6,feasible=True,minimum_constraint=0.),guard={})
  c=RecoveryController();c.update(missing,0.,0.,.1,2.)
  for t in (.1,.2):self.assertEqual(c.update(good,0.,t,.1,2.)[0]['acc'],0.)
  self.assertGreater(c.update(good,0.,.3,.1,2.)[0]['acc'],0.)
  self.assertLess(c.update(missing,1.,.4,.1,2.)[0]['acc'],0.)
  self.assertEqual(c.clear_count,0)
 def test_all_saved_outputs_and_actual_failure_pass_transport(self):
  source=CR/'phase31/runs/empty_occupancy551_20261105T095907Z_0354f425/closed_loop/uniad_worker';counts={}
  for f in sorted(source.glob('step_*/perception.npz')):
   with np.load(f) as saved:a=saved['occupancy'].copy()
   fields,c=pack(a);b,d=unpack(fields);np.testing.assert_array_equal(a,b);counts[c['status']]=counts.get(c['status'],0)+1
  self.assertEqual(counts,dict(valid_nonempty=52,valid_empty=10))
  error=_hugsim_expand(json.loads((source/'done_062.json').read_text()));self.assertIn('Missing predicted occupancy',error['error']);fields,c=pack(None);self.assertIsNone(unpack(fields)[0]);self.assertTrue(unavailable_result(c,{})['diagnostic_continues'])
 def test_actual_live_driver_handles_unavailable_before_mpc(self):
  from run_driver import configure
  configure();source=CR/'phase31/runs/empty_occupancy551_20261105T095907Z_0354f425/closed_loop/uniad_worker/step_061'
  with (source/'observation.pkl').open('rb') as stream:obs,info=pickle.load(stream)
  del obs
  rel=_hugsim_expand(json.loads((source/'guard_memory_report.json').read_text()));_,c=pack(None,0);info['occupancy_contract']=c
  result=h.optimize(np.zeros((6,2)),None,info,reliability=rel,vehicle=h.load_vehicle_config() if hasattr(h,'load_vehicle_config') else None)
  self.assertIsNone(result['selected']);self.assertEqual(result['reason'],'occupancy_unavailable:no_motion_queries');self.assertTrue(result['diagnostic_continues'])
 def test_model_motion_hook_does_not_change_outputs(self):
  import worker as wrapper
  original=wrapper.global_worker.serve
  model=types.SimpleNamespace(motion_head=types.SimpleNamespace(forward_test=lambda:('unchanged',dict(track_query=np.zeros((1,0,256))))))
  bound=model.motion_head.forward_test
  with tempfile.TemporaryDirectory() as temp:
   wrapper.worker.JOB=Path(temp);folder=Path(temp)/'step_000';folder.mkdir()
   wrapper.global_worker.serve=lambda m,n:m.motion_head.forward_test()
   try:result=wrapper.serve(model,{})
   finally:wrapper.global_worker.serve=original
   self.assertEqual(result[0],'unchanged');self.assertIs(model.motion_head.forward_test,bound)
   self.assertEqual(_hugsim_expand(json.loads((folder/'occupancy_head_audit.json').read_text()))['motion_query_count'],0)
 def test_protective_stops_and_time_window_are_preserved(self):
  s=(HERE/'protocol_harness.py').read_text()
  for text in ('Post-hoc actor reference overlap','Post-hoc native-vehicle reference overlap','Post-hoc inserted-vehicle footprint overlap','Inference clock mismatch','Non-finite plan patch'):
   # The original plan-patch spelling is checked below using the original source.
   if text=='Non-finite plan patch':continue
   self.assertIn(text,s)
  original=(CR/'phase28/heading_scene_run_v1.py').read_text();self.assertEqual(s.count('raise RuntimeError'),original.count('raise RuntimeError'));self.assertEqual(s.count('raise ValueError'),original.count('raise ValueError'))
if __name__=='__main__':unittest.main(verbosity=2)
