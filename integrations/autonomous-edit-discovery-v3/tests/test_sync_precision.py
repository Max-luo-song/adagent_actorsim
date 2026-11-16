
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

import ast,copy,tempfile,unittest
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from common_flow import ROOT,OUT,read,write,asset,metrics,UnscoredEpisode
from sync_verifier import exact_runtime_values,ReplaceAssertions

class SyncPrecisionTests(unittest.TestCase):
 def failed_work(self):return OUT/'tasks/llm_s551_b003_e02_037db15d/global/cases/cutin551/on'
 def test_actual_failed_snapshot_is_exact_renderer_roundtrip(self):
  work=self.failed_work()
  with np.load(work/'scenario/target_track.npz') as expected,np.load(work/'B/actor_tracks.npz') as actual:
   for field,key in [('rendered_trans','RigidNodes__instances_trans'),('rendered_quats','RigidNodes__instances_quats')]:exact_runtime_values(actual[key][:,3],expected[field])
   self.assertGreater(float(np.abs(actual['RigidNodes__instances_trans'][:,3]-expected['rendered_trans']).max()),1e-5)
 def test_material_change_nonfinite_and_wrong_dtype_still_fail(self):
  expected=np.array([385.08754892310407],dtype=np.float64);actual=expected.astype(np.float32);exact_runtime_values(actual,expected)
  changed=np.nextafter(actual,np.float32(np.inf))
  with self.assertRaises(AssertionError):exact_runtime_values(changed,expected)
  with self.assertRaises(ValueError):exact_runtime_values(np.array([float('nan')],dtype=np.float32),expected)
  with self.assertRaises(ValueError):exact_runtime_values(actual.astype(np.float64),expected)
 def test_real_parent_guard_fails_and_repaired_guard_preserves_all_other_checks(self):
  source=ROOT/'integrations/light-tracking-curvature-v15-r2/pair_diagnostics.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_actor_sync')
  module=ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]);namespace=dict(np=np,Path=Path,read=read,write=write,asset=asset)
  exec(compile(module,str(source),'exec'),namespace);parent=namespace['verify_actor_sync']
  repair=ReplaceAssertions();fixed=repair.visit(copy.deepcopy(module));ast.fix_missing_locations(fixed);self.assertEqual(repair.count,2)
  namespace['_exact_runtime_values']=exact_runtime_values;exec(compile(fixed,str(source),'exec'),namespace);candidate=namespace['verify_actor_sync']
  work=self.failed_work()
  with np.load(work/'B/actor_tracks.npz') as data:tracks={k:data[k].copy() for k in data.files}
  state={k:tracks['RigidNodes__'+k] for k in ('instances_trans','instances_quats','instances_size','instances_fv')}
  model=SimpleNamespace(**state);session=SimpleNamespace(backend=SimpleNamespace(trainer=SimpleNamespace(models={'RigidNodes':model}),env=SimpleNamespace(models={'RigidNodes':state})),vehicles=SimpleNamespace(programs={3:None}))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'scenario').mkdir();(root/'B').mkdir();write(root/'plan.json',read(work/'plan.json'))
   (root/'scenario/target_track.npz').write_bytes((work/'scenario/target_track.npz').read_bytes());(root/'B/actor_tracks.npz').write_bytes((work/'B/actor_tracks.npz').read_bytes())
   with self.assertRaises(AssertionError):parent(root,tracks,session,{'activation_frame':15})
   result=candidate(root,tracks,session,{'activation_frame':15});self.assertTrue(result['passed']);self.assertTrue(all(result['checks'].values()))
 def test_zero_control_failure_retains_root_cause_without_surrogate_scores(self):
  with self.assertRaises(UnscoredEpisode) as raised:metrics(self.failed_work(),'B')
  self.assertEqual(raised.exception.controls,0);self.assertEqual(raised.exception.error_stage,'scene_load');self.assertIn('AssertionError',raised.exception.source_error)
 def test_retry_uses_new_output_and_retains_original_failure(self):
  import flow
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);work=root/'on';work.mkdir();fresh=work/'infra_retries/attempt_001';write(root/'task.json',{});write(root/'policy.json',{});write(work/'failure.json',{'status':'failed'});write(fresh/'completion.json',{})
   failure=asset(work/'failure.json');write(work/'active_retry.json',dict(task=asset(root/'task.json'),policy=asset(root/'policy.json'),case='cutin551',work=str(fresh),preserved_failure={'report':failure}))
   with patch.object(flow.subprocess,'run'),patch.object(flow,'metrics',return_value={'result':'fresh'}) as result:
    self.assertEqual(flow.probe({},root/'task.json',root/'policy.json',work,'cutin551'),{'result':'fresh'});result.assert_called_once_with(fresh.resolve(),'B')
   self.assertEqual(asset(work/'failure.json'),failure)
