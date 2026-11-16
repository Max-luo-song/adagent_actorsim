
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import ast,copy,importlib.util,sys,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];ROOT=LOCAL.parents[1];CODE=ROOT/'integrations/light-tracking-curvature-v15-r2'
sys.path[:0]=[str(LOCAL),str(CODE),str(CODE/'tests'),str(ROOT/'integrations/light-map551-v4'),str(ROOT/'integrations/light-cutin-diagnostic')]
from protocol import CASES,SHARED,evaluate_pair,overall,validate_contracts

def arm(mode='off'):
 c={k:'shared' for k in SHARED};c.update(scene=551,edited=False,patch='on',comparison_version='v13' if mode=='off' else 'v15',fresh_history=True,score_kind='executed_path')
 return dict(contract=c,status='completed',score=dict(hdscore=.5,pdms=1.),collision=False,map_departure=False,unresolved_stop=False,progress_m=50.,stopped_s=0.)
class ProtocolTests(unittest.TestCase):
 def test_healthy_unchanged_original_passes_only_its_pair(self):
  r=evaluate_pair('original551',arm(),arm('on'));self.assertTrue(r['passed']);self.assertFalse(overall({'original551':r})['global_patch_validated'])
 def test_target_requires_real_improvement(self):
  self.assertFalse(evaluate_pair('cutin551',arm(),arm('on'))['passed'])
  a=arm('on');a['score']['hdscore']=.55;self.assertTrue(evaluate_pair('cutin551',arm(),a)['passed'])
 def test_every_shared_input_is_bound(self):
  for key in SHARED:
   a=arm('on');a['contract'][key]='changed'
   with self.assertRaises(ValueError):evaluate_pair('original551',arm(),a)
 def test_raw_planning_or_replayed_history_cannot_replace_control_arm(self):
  for key,value in [('score_kind','raw_uniad'),('fresh_history',False),('patch','off')]:
   a=arm('on');a['contract'][key]=value
   with self.assertRaises(ValueError):evaluate_pair('original551',arm(),a)
 def test_missing_failed_or_unscored_never_passes(self):
  self.assertFalse(evaluate_pair('original551',arm(),None)['passed'])
  for key,value in [('score',None),('status','resource_limited'),('status','failed')]:
   a=arm('on');a[key]=value;self.assertEqual(evaluate_pair('original551',arm(),a)['status'],'invalid')
 def test_nan_is_rejected(self):
  a=arm('on');a['score']['hdscore']=float('nan');self.assertEqual(evaluate_pair('original551',arm(),a)['status'],'invalid')
 def test_safety_or_unresolved_stop_overrides_score_gain(self):
  for key in ('collision','map_departure','unresolved_stop'):
   a=arm('on');a[key]=True;a['score']['hdscore']=.8;self.assertFalse(evaluate_pair('cutin551',arm(),a)['passed'])
 def test_each_original_scene_must_preserve_progress_stop_and_score(self):
  for key,value in [('progress_m',40.),('stopped_s',2.),('score',dict(hdscore=.48,pdms=1.)),('score',dict(hdscore=.5,pdms=.98))]:
   a=arm('on');a[key]=value;self.assertFalse(evaluate_pair('original551',arm(),a)['passed'])
 def test_bad_normal_baseline_is_not_nonregression_certification(self):
  a=arm();a['collision']=True;self.assertEqual(evaluate_pair('original551',a,arm('on'))['status'],'invalid')
 def test_averaging_cannot_hide_failed_case(self):
  a={k:dict(status='passed',passed=True) for k in CASES};a['original552']=dict(status='failed',passed=False)
  self.assertFalse(overall(a)['global_patch_validated']);self.assertEqual(overall(a)['status'],'failed')
 def test_score_annotation_preserves_values_and_binds_actual_patch_mode(self):
  spec=importlib.util.spec_from_file_location('_validation_score_metadata',LOCAL/'pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  source=dict(scores=dict(executed_path=dict(hdscore=dict(value=.5))),ground_truth_used_by_optimizer=True)
  off=m.annotate_score(source,'off');on=m.annotate_score(source,'on')
  self.assertFalse(off['ground_truth_used_by_optimizer']);self.assertTrue(on['ground_truth_used_by_optimizer'])
  self.assertEqual(off['scores'],source['scores']);self.assertEqual(on['scores'],source['scores']);self.assertTrue(source['ground_truth_used_by_optimizer'])
 def test_cross_scene_holes_are_explicit(self):
  a={k:dict(status='passed',passed=True) if v['ready'] else dict(status='unavailable',passed=False) for k,v in CASES.items()}
  self.assertEqual(overall(a)['status'],'pending');self.assertFalse(overall(a)['global_patch_validated'])
class ControllerTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from common import configure
  configure()
  import controller_switch
  cls.switch=controller_switch
 def test_on_is_the_exact_frozen_patch_class(self):
  from test_repair import V,mapping
  import joint_safety
  self.switch.MODE='on';r=self.switch.JointReference(V,mapping());self.assertIs(type(r),joint_safety.JointReference)
 def test_off_preserves_current_reference_and_nominal_action(self):
  import numpy as np
  from test_repair import V,mapping,Motion,intent,observation
  self.switch.MODE='off';m=Motion(.1,2.7,4.);r=self.switch.JointReference(V,mapping());a,inf=observation(m,0);raw=intent(m);saved=raw.copy()
  out,s=r.process(raw,a,{},inf,m);np.testing.assert_array_equal(out,saved);np.testing.assert_array_equal(raw,saved);self.assertIsNone(r.selected_states)
  nominal=dict(acc_mps2=-.7,steer_rate_radps=.03);action,g=r.admit(m,nominal);self.assertEqual(action,nominal);self.assertFalse(g['external_patch_executed'])
 def test_runtime_changes_only_one_controller_import(self):
  source=(CODE/'runtime.py').read_text();expected=ast.parse(source)
  for n in ast.walk(expected):
   if isinstance(n,ast.ImportFrom) and n.module=='joint_safety':n.module='controller_switch'
  self.assertEqual(ast.dump(self.switch.rewrite_runtime(source)),ast.dump(expected))
  with self.assertRaises(ValueError):self.switch.rewrite_runtime('x=1')
 def run_fixture(self,mode):
  # Reuse actual runtime + actual iLQR/Motion, with explicitly mocked scene/NN.
  spec=importlib.util.spec_from_file_location('_frozen_v11_fixture',CODE/'tests/test_cutin.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
  ps=importlib.util.spec_from_file_location('_validation_pipeline_fixture',LOCAL/'pipeline.py');pipeline=importlib.util.module_from_spec(ps);ps.loader.exec_module(pipeline)
  fixture.runtime=pipeline.load_runtime(mode)
  fixture.CutinTests('test_real_tracker_motion_common_history_and_absolute_actor_clock').test_real_tracker_motion_common_history_and_absolute_actor_clock()
 def test_actual_runtime_original_and_cutin_off_with_mocked_gpu(self):self.run_fixture('off')
 def test_actual_runtime_original_and_cutin_on_with_mocked_gpu(self):self.run_fixture('on')
if __name__=='__main__':unittest.main(verbosity=2)
