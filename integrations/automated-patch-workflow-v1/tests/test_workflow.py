
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

import copy,json,sys,tempfile,unittest
pass  # Path is provided by the portable boundary above
from unittest.mock import patch
LOCAL=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LOCAL));import workflow as w
class WorkflowTests(unittest.TestCase):
 def test_identifiers_cannot_be_shell_or_paths(self):
  for s in ('../other','/tmp/x','foo;bar','x$(id)',''):
   with self.assertRaises(ValueError):w.identity(s)
 def test_relative_target_is_confined(self):
  for s in ('../joint_safety.py','/tmp/file'):
   with self.assertRaises(ValueError):w.path_under(Path('/tmp/candidate'),s)
 def test_symlink_target_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').write_text('x');(p/'link').symlink_to(p/'a')
   with self.assertRaises(ValueError):w.path_under(p,'link')
 def test_safety_methods_cannot_be_replaced(self):
  s=(w.BASE/'joint_safety.py').read_text()
  w.safety_equal(s,s)
  bad=s.replace('if not len(boxes):return dict(passed=True','if not len(boxes):return dict(passed=False',1)
  with self.assertRaises(ValueError):w.safety_equal(s,bad)
 def test_preview_and_physical_policy_cannot_be_weakened(self):
  s=(w.BASE/'joint_safety.py').read_text()
  with self.assertRaises(ValueError):w.safety_equal(s,s.replace('max_preview_s=12.','max_preview_s=3.'))
 def test_obstacle_memory_is_unchanged_except_reset_attributes(self):
  s=(w.BASE/'risk_memory.py').read_text();w.memory_equal(s,s)
  w.memory_equal(s,s.replace('self.previous_target_speed=None','self.previous_target_speed=None;self.intent_clear_since=None'))
  with self.assertRaises(ValueError):w.memory_equal(s,s.replace('if not t[\'confirmed\']:continue','if True:continue'))
 def test_asset_mutation_breaks_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('a');a=w.asset(p);w.verify(a);p.write_text('b')
   with self.assertRaises(ValueError):w.verify(a)
 def test_successful_stage_is_idempotent(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);count=root/'count';cmd=[sys.executable,'-c',"from pathlib import Path;p=Path("+repr(str(count))+");p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"]
   with patch.object(w,'verify_candidate',return_value={}):
    w.run_registered(root,'smoke',cmd);w.run_registered(root,'smoke',cmd)
    self.assertEqual(count.read_text(),'1')
    with self.assertRaises(ValueError):w.run_registered(root,'smoke',cmd+['changed'])
 def test_rejected_first_proposal_cannot_be_reapplied(self):
  with self.assertRaises(ValueError):w.decisions(w.STORE/'rounds/v11_to_v12_intent_recovery')
 def test_uncertain_diagnosis_never_reads_or_applies_patch(self):
  r=w.STORE/'rounds/v11_to_v12_intent_recovery_r2';real=w.read
  def pending(path):
   if Path(path).name=='patch_instruction.json':raise AssertionError('Patch should not be consumed')
   x=real(path)
   if Path(path).name=='failure_reason.json':x=copy.deepcopy(x);x['decision']='diagnostics_required'
   return x
  with patch.object(w,'read',side_effect=pending),patch.object(w,'update'):
   with self.assertRaises(w.DiagnosticsPending):w.decisions(r)
 def test_completed_cycle_does_not_rerun_frozen_cpu_artifacts(self):
  r=w.STORE/'rounds/v11_to_v12_intent_recovery_r2'
  with patch.object(w,'run_registered',side_effect=AssertionError('Must not rerun completed stages')):
   self.assertIn(w.cycle(r)['stage'],('awaiting_gpu_validation','pending_evidence'))
 def test_v11_increment_requires_an_actual_verified_v12_target(self):
  r=w.STORE/'rounds/v11_to_v12_intent_recovery_r2'
  result=w.incremental_target(r,{},dict(details=dict(cutin551=dict(on=None))))
  self.assertEqual(result['status'],'pending');self.assertFalse(result['accepted'])
 def test_incomplete_v12_cannot_pass_incremental_gate(self):
  r=w.STORE/'rounds/v11_to_v12_intent_recovery_r2'
  result=w.incremental_target(r,{},dict(details=dict(cutin551=dict(on=dict(status='resource_limited',score=None)))))
  self.assertEqual(result['status'],'invalid');self.assertFalse(result['accepted'])
 def test_nonfinite_json_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('{"score":NaN}')
   with self.assertRaises(ValueError):w.read(p)
 def test_llm_decision_requires_bound_evidence_and_full_acceptance(self):
  r=w.STORE/'rounds/v11_to_v12_intent_recovery_r2'
  e,f,p,edit=w.decisions(r)
  self.assertEqual(p['base_version'],'v11');self.assertEqual(p['candidate_version'],'v12');self.assertEqual(edit['simulation']['profile'],'risk_provenance_v11_diagnostic')
  real_read=w.read
  def weakened(path):
   data=real_read(path)
   if Path(path).name=='patch_instruction.json':data=copy.deepcopy(data);data['acceptance']['cross_scene_nonregression_required']=False
   return data
  with patch.object(w,'read',side_effect=weakened):
   with self.assertRaises(ValueError):w.decisions(r)
if __name__=='__main__':unittest.main(verbosity=2)
