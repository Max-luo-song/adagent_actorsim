
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

import ast,copy,importlib.util,json,sys,unittest
pass  # Path is provided by the portable boundary above
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL))
from gates import compare,values,replay_gate
class ScoreGateTests(unittest.TestCase):
 def episode(self):return dict(status='completed',controls=35,fresh_history=True,executed_path_scored=True,safe=True,scores={k:{'value':.5} for k in ['hdscore','pdms','rc']})
 def test_equal_or_better_all_metrics_pass(self):
  a=self.episode();b=copy.deepcopy(a);b['scores']['hdscore']['value']+=.02
  self.assertTrue(compare(a,b,35)['passed'])
 def test_one_scene_metric_drop_cannot_be_averaged_away(self):
  a=self.episode();b=copy.deepcopy(a);b['scores']['hdscore']['value']+=.2;b['scores']['rc']['value']-=.001
  self.assertFalse(compare(a,b,35)['passed'])
 def test_null_bounds_and_nonfinite_scores_not_promoted(self):
  for value in [None,float('nan'),float('inf'),True,{'lower':.1,'upper':1.}]:
   a=self.episode();a['scores']['hdscore']=value
   self.assertIsNone(values(a['scores']));self.assertFalse(compare(self.episode(),a,35)['comparable'])
 def test_short_failure_replay_unsafe_or_unscored_never_pass(self):
  for k,v in [('status','terminated'),('controls',21),('fresh_history',False),('executed_path_scored',False),('safe',False)]:
   b=self.episode();b[k]=v;self.assertFalse(compare(self.episode(),b,35)['passed'])
 def test_empty_or_partial_replay_cannot_pass(self):
  self.assertFalse(replay_gate({}));self.assertFalse(replay_gate({'172_v15':{'rows':[]}}))
 def test_known_old_552_rejection_is_visible_not_success(self):
  r={k:{'rows':[{}]*{'172_v15':22,'172_native':36,'551_original':189,'551_cutin':189,'552_v15':25}[k],'same_observation_counterfactual_only':True,'no_action_frames':([24] if k=='552_v15' else [])} for k in ['172_v15','172_native','551_original','551_cutin','552_v15']}
  self.assertTrue(replay_gate(r));r['172_native']['no_action_frames']=[9];self.assertFalse(replay_gate(r))
class BindingTests(unittest.TestCase):
 def test_repair_attachment_matches_frozen_scene_adaptation(self):
  root=LOCAL.parents[1];cross=root/'integrations/cross-scene-validation-v2';repair=root/'integrations/box-validity-repair-v1'
  sys.path.insert(0,str(repair))
  from live_contract import rewrite
  s=importlib.util.spec_from_file_location('frozen_cross_adapter',cross/'bindings.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  source=(root/'integrations/light-execution-horizon-v17-r2/runtime.py').read_text()
  for scene,command,count in [(172,1,28),(552,0,11)]:
   tree,changes=m.runtime_tree(source,scene,command,count);repaired=rewrite(ast.unparse(tree));text=ast.unparse(repaired)
   self.assertEqual(text.count('attach_box_observation(perception, stepdir, info, spec.vehicle)'),1);self.assertIn('scene='+str(scene),text.replace(' ',''));compile(repaired,'test_bound_runtime','exec')
 def test_fixed_three_arm_cases_and_no_resident_write(self):
  source=(LOCAL/'pipeline.py').read_text();self.assertIn("ARMS=('off','v15','repair')",source)
  self.assertNotIn('accepted_patch.json',source);self.assertNotIn("write(P26/'global_patch.json'",source)
if __name__=='__main__':unittest.main()
